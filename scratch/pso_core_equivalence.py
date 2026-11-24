"""PSO-core equivalence check: calypso's CALYPSO vs edaptive's PSO, fixed CPs.

Run from the repo root:

    python scratch/pso_core_equivalence.py [--seeds 30] [--workers 10]

Both PSOs run with the same fixed CPs at every iteration (w = 0.7298,
c_1 = c_2 = 1.49618), velocity clamping to the full range (calypso's ``--vc``,
edaptive's ``V_max_fraction = 1.0``), no policy, 30 particles, 30-D, 5000
iterations, on the 5 functions of ``sac_sapso_equivalence.py``, with its seeds:
run ``s`` of a function uses ``trial_seed(name, 30, s, base=seed_base)``
(``function_seeds``), so problems get distinct seeds. calypso runs through
``CALYPSO.iter`` (as ``calypsoEnv`` does), in float64.

Per run it records the final gbest and, after iterations 1, 50, 500, 2500 and
5000 (1-based; edaptive's step t is calypso's iteration t + 1), the swarm
diversity (mean distance to the centroid), the mean |V| / (ub - lb) over particles
and dimensions, and the infeasible proportion. Each quantity is compared across
seeds with a two-sided Mann-Whitney U test; Holm adjusts over all tests in the
script. Writes ``results/sac_training/pso_core_equivalence.json``.
"""

import argparse
import json
import os
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from scipy.stats import mannwhitneyu

REPO = Path(__file__).resolve().parents[1]
CALYPSO_ROOT = REPO / "calypso"
sys.path.insert(0, str(CALYPSO_ROOT))

from sac_sapso_equivalence import function_seeds  # noqa: E402

from edaptive.optimizers.pso import PSO, swarm_diversity  # noqa: E402
from edaptive.problems.ela_benchmark_suite import (  # noqa: E402
    BBOB_FID17_IID2,
    Brown,
    GeneralizedEggCrate,
    Qing,
    Rosenbrock,
)

FUNCTIONS = [Rosenbrock, Qing, GeneralizedEggCrate, Brown, BBOB_FID17_IID2]
W, C_1, C_2 = 0.7298, 1.49618, 1.49618
N_S, N_X, MAX_ITER = 30, 30, 5000
CHECKPOINTS = (1, 50, 500, 2500, 5000)  # 1-based iterations
SUMMARIES = ("diversity", "mean_abs_V", "infeasible")


def summaries(X, V, lb, ub):
    return {
        "diversity": float(swarm_diversity(X)),
        "mean_abs_V": float(np.mean(np.abs(V / (ub - lb)))),
        "infeasible": float(np.mean(np.any((X < lb) | (X > ub), axis=1))),
    }


def run_calypso(problem_type, seed):
    os.chdir(CALYPSO_ROOT)  # calypso's const.py paths
    from scripts.pso.calypso import CALYPSO

    problem = problem_type(dimensions=N_X)
    lb, ub = problem.lower, problem.upper
    pso = CALYPSO(
        exp="pso_core_equivalence",
        run=0,
        seed=seed,
        loss=problem,
        lim=(lb, ub),
        name=problem_type.__name__,
        N=N_S,
        D=N_X,
        maxIterCal=MAX_ITER,
        velocityClamping=True,
        rl=True,
        seterr="ignore",
        dtype=np.float64,
    )
    control = {"w": W, "c1": C_1, "c2": C_2}
    record = {}
    for i in range(1, MAX_ITER + 1):
        pso.iter(i=i, j=1, controlParam=control, funcReward=0)
        if i in CHECKPOINTS:
            record[i] = summaries(pso.P[i], pso.V[i], lb, ub)
    return float(pso.L_social_best), record


def run_edaptive(problem_type, seed):
    problem = problem_type(dimensions=N_X)
    lb, ub = problem.lower, problem.upper
    pso = PSO(
        problem=problem,
        rng=np.random.default_rng(seed),
        n_s=N_S,
        w=W,
        c_1=C_1,
        c_2=C_2,
        V_max_fraction=1.0,
    )
    record = {}
    for t in range(MAX_ITER):
        pso.step(t)
        if t + 1 in CHECKPOINTS:
            record[t + 1] = summaries(pso.X, pso.V, lb, ub)
    return float(pso.f_Y_hat), record


def job(args):
    side, k, seed = args
    run = run_calypso if side == "calypso" else run_edaptive
    return side, k, seed, run(FUNCTIONS[k], seed)


def mwu(a, b):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    both = np.concatenate([a, b])
    p = 1.0 if np.ptp(both) == 0 else float(mannwhitneyu(a, b).pvalue)
    return {
        "median_calypso": float(np.median(a)),
        "median_edaptive": float(np.median(b)),
        "mean_calypso": float(np.mean(a)),
        "mean_edaptive": float(np.mean(b)),
        "p": p,
    }


def holm(p_values):
    p = np.asarray(p_values, dtype=float)
    order = np.argsort(p, kind="stable")
    adjusted = np.empty_like(p)
    running = 0.0
    for i, index in enumerate(order):
        running = max(running, min(1.0, (len(p) - i) * p[index]))
        adjusted[index] = running
    return adjusted


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seeds", type=int, default=30)
    parser.add_argument("--seed-base", type=int, default=5_000_000)
    parser.add_argument("--workers", type=int, default=10)
    args = parser.parse_args()

    seeds_by_function = [
        function_seeds(f, N_X, args.seed_base, args.seeds) for f in FUNCTIONS
    ]
    jobs = [
        (side, k, s)
        for k in range(len(FUNCTIONS))
        for s in seeds_by_function[k]
        for side in ("calypso", "edaptive")
    ]
    results = {}
    with Pool(args.workers, maxtasksperchild=1) as pool:
        for side, k, seed, result in pool.imap_unordered(job, jobs):
            results[(side, k, seed)] = result

    report = {}
    for k, problem_type in enumerate(FUNCTIONS):
        tests = {}
        runs = {
            side: [results[(side, k, s)] for s in seeds_by_function[k]]
            for side in ("calypso", "edaptive")
        }
        tests["f_final"] = mwu(
            [f for f, _ in runs["calypso"]], [f for f, _ in runs["edaptive"]]
        )
        for c in CHECKPOINTS:
            for q in SUMMARIES:
                tests[f"{q}@{c}"] = mwu(
                    [r[c][q] for _, r in runs["calypso"]],
                    [r[c][q] for _, r in runs["edaptive"]],
                )
        report[problem_type.__name__] = tests

    keys = [(f, q) for f in report for q in report[f]]
    for (f, q), p in zip(keys, holm([report[f][q]["p"] for f, q in keys]), strict=True):
        report[f][q]["p_holm"] = float(p)

    for name, tests in report.items():
        print(f"\n{name}")
        print(f"  {'quantity':<20}{'med calypso':>14}{'med edaptive':>14}"
              f"{'p':>9}{'p Holm':>9}")
        for q, t in tests.items():
            print(f"  {q:<20}{t['median_calypso']:>14.6g}{t['median_edaptive']:>14.6g}"
                  f"{t['p']:>9.4f}{t['p_holm']:>9.4f}")

    out = REPO / "results" / "sac_training" / "pso_core_equivalence.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as file:
        json.dump(
            {
                "seed_base": args.seed_base,
                "seeds": {
                    f.__name__: seeds
                    for f, seeds in zip(FUNCTIONS, seeds_by_function, strict=True)
                },
                "cps": {"w": W, "c_1": C_1, "c_2": C_2},
                "n_s": N_S,
                "n_x": N_X,
                "max_iterations": MAX_ITER,
                "checkpoints": list(CHECKPOINTS),
                "n_tests": len(keys),
                "functions": report,
            },
            file,
            indent=2,
        )
    print(f"\n{len(keys)} tests; wrote {out}")


if __name__ == "__main__":
    main()
