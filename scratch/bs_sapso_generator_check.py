"""Generator diagnostic for Known issue 17 (CLAUDE.md): is the BBOB_FID17_IID2 shift a
random-generator effect?

Run from the repo root:

    nohup taskset -c 0,2,4,6 python scratch/bs_sapso_generator_check.py \\
        > results/bs_sapso_equivalence_seed7000000/generator_check.log 2>&1 &

Design, fixed before the run: BBOB_FID17_IID2, improvement, update_freq 201 (calypso's
vc_fixed200_improve30), 30-D, 5000 iterations, seed bases 12-15 million with 125
seeds each (500 runs per arm), trial_seed(name, 30, s, base) per run. Three arms:

- calypso: CALYPSO as in bs_sapso_equivalence.py (MT19937 through numpy's legacy
  RandomState, calypso's draw layout);
- BSSAPSO-MT19937: np.random.Generator(np.random.MT19937(seed)) swapped in for
  np.random.default_rng while the solver is built (edaptive's draw layout);
  diagnostic only, BSSAPSO's default is unchanged;
- BSSAPSO-PCG64: BSSAPSO as normal.

Two-sided Mann-Whitney U on the final gbest for each pair, with medians. Decision
rule: BSSAPSO-MT19937 vs BSSAPSO-PCG64 at p < 0.01 is reported as a generator
effect; otherwise the shift is treated as chance. Writes
results/bs_sapso_equivalence_seed7000000/generator_check.json.
"""

import json
import sys
import time
from itertools import combinations
from multiprocessing import Pool

import numpy as np
from scipy.stats import mannwhitneyu

sys.path.insert(0, "scratch")
import bs_sapso_bbob_trajectories as inv  # noqa: E402

import edaptive.core.solver as core_solver  # noqa: E402
from edaptive.solvers.pso.bs_sapso import BSSAPSO  # noqa: E402
from edaptive.utils.seeding import trial_seed  # noqa: E402

NAME = "BBOB_FID17_IID2"
BASES = (12_000_000, 13_000_000, 14_000_000, 15_000_000)
SEEDS_PER_BASE = 125
ARMS = ("calypso", "BSSAPSO-MT19937", "BSSAPSO-PCG64")
OUT = "results/bs_sapso_equivalence_seed7000000/generator_check.json"


def bssapso_mt19937(seed):
    original = core_solver.np.random.default_rng
    core_solver.np.random.default_rng = lambda s: np.random.Generator(
        np.random.MT19937(s)
    )
    try:
        solver = BSSAPSO(
            problem=inv.problem(NAME),
            config=inv.eq.edaptive_config("improvement", 201, inv.N_S),
            rnd_seed=seed,
        )
    finally:
        core_solver.np.random.default_rng = original
    _, history = solver.run(inv.T)
    return float(np.min(history["f_best"]))


def task(job):
    arm, base, s = job
    seed = trial_seed(NAME, inv.N_X, s, base=base)
    if arm == "calypso":
        f_final = inv.bs_calypso(NAME, seed, s)[-1]
    elif arm == "BSSAPSO-MT19937":
        f_final = bssapso_mt19937(seed)
    else:
        f_final = inv.bs_edaptive(NAME, seed)[-1]
    return (arm, base, s), f_final


def main():
    started = time.perf_counter()
    jobs = [(arm, b, s) for b in BASES for s in range(SEEDS_PER_BASE) for arm in ARMS]
    jobs.sort(key=lambda job: job[0] != "calypso")  # slowest first
    results = {}
    with Pool(4, maxtasksperchild=40) as pool:
        for n, (key, value) in enumerate(pool.imap_unordered(task, jobs), 1):
            results[key] = value
            if n % 60 == 0 or n == len(jobs):
                elapsed = time.perf_counter() - started
                print(f"{n}/{len(jobs)} runs, {elapsed:.0f} s", flush=True)

    final = {arm: [v for (a, _, _), v in results.items() if a == arm] for arm in ARMS}
    pairs = {}
    for a, b in combinations(ARMS, 2):
        pairs[f"{a} vs {b}"] = {
            f"median {a}": float(np.median(final[a])),
            f"median {b}": float(np.median(final[b])),
            "p": float(mannwhitneyu(final[a], final[b]).pvalue),
        }
    decisive = pairs["BSSAPSO-MT19937 vs BSSAPSO-PCG64"]["p"]
    verdict = "generator effect" if decisive < 0.01 else "chance"
    output = {
        "design": {
            "function": NAME,
            "selection": "improvement",
            "update_freq": 201,
            "max_iter": inv.T,
            "seed_bases": list(BASES),
            "seeds_per_base": SEEDS_PER_BASE,
            "runs_per_arm": {arm: len(v) for arm, v in final.items()},
            "decision_rule": "BSSAPSO-MT19937 vs BSSAPSO-PCG64 at p < 0.01 = "
            "generator effect, else chance",
        },
        "pairs": pairs,
        "verdict": verdict,
        "per_run": {f"{a}/{b}/{s}": v for (a, b, s), v in results.items()},
        "wall_seconds": time.perf_counter() - started,
    }
    with open(OUT, "w") as file:
        file.write(json.dumps(output, indent=1) + "\n")
    for pair, r in pairs.items():
        medians = "  ".join(f"{k} {v:.3f}" for k, v in r.items() if k != "p")
        print(f"{pair:40s} {medians}  p = {r['p']:.4f}")
    print("verdict:", verdict)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
