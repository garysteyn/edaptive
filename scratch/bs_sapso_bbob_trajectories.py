"""BBOB_FID17_IID2 diagnostics for Known issue 17 (CLAUDE.md): problem checks and gbest
trajectories, calypso vs edaptive.

Run from the repo root:

    nohup taskset -c 0,2,4,6 python scratch/bs_sapso_bbob_trajectories.py \\
        > results/bs_sapso_equivalence_seed7000000/bbob_investigation.log 2>&1 &

1. Problem checks: bounds, batched vs row-by-row evaluation, calypso's getLoss
   (feasible rows only, +inf elsewhere) vs the problem, and no state left in COCO
   by evaluating infeasible points.
2. Median gbest at checkpoints over 60 runs per side (seed bases 6 and 7 million,
   30 seeds each): BSSAPSO improvement (update_freq 201) vs calypso's
   vc_fixed200_improve30, and the PSO core with fixed CPs (calypso's 'constant'
   baseline vs BasicSolver) on BBOB_FID17_IID2 and Rosenbrock. Two-sided MWU per
   checkpoint.

Writes results/bs_sapso_equivalence_seed7000000/bbob_investigation.json.
"""
import contextlib, io, json, os, sys
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(v, "1")
from multiprocessing import Pool
from pathlib import Path
import numpy as np
from scipy.stats import mannwhitneyu

REPO = Path("/workspaces/claude-sandbox/edaptive")
sys.path.insert(0, str(REPO / "calypso"))
sys.path.insert(0, str(REPO / "scratch"))
import bs_sapso_equivalence as eq  # noqa: E402
from edaptive.core.config import SolverConfig
from edaptive.optimizers.pso import PSO
from edaptive.problems.ela_benchmark_suite import BBOB_FID17_IID2, Rosenbrock
from edaptive.solvers.basic_solver import BasicSolver
from edaptive.solvers.pso.bs_sapso import BSSAPSO
from edaptive.utils.seeding import trial_seed

N_X, N_S, T = 30, 30, 5000
CHECKPOINTS = [0, 1, 10, 50, 100, 200, 500, 1000, 2000, 3000, 4000, 5000]
FUNCS = {"BBOB_FID17_IID2": BBOB_FID17_IID2, "Rosenbrock": Rosenbrock}
_P = {}

def problem(name):
    if name not in _P:
        _P[name] = FUNCS[name](N_X)
    return _P[name]

def bs_calypso(name, seed, s):
    from scripts.pso.calypso import CALYPSO
    p = problem(name)
    kw, _ = eq.calypso_kwargs(eq.execute_argv("improve", 200, N_X, N_S, T), s, p, (p.lower, p.upper), name)
    c = CALYPSO(**kw, seed=seed, savePath=False)
    with contextlib.redirect_stdout(io.StringIO()):
        _, f, _ = c.iterate()
    return list(map(float, c.data["globalBests"])) + [float(f)]

def bs_edaptive(name, seed):
    cfg = eq.edaptive_config("improvement", 201, N_S)
    _, h = BSSAPSO(problem=problem(name), config=cfg, rnd_seed=seed).run(T)
    return list(map(float, h["f_best"]))

W, C = 0.729844, 1.496180  # calypso's 'constant' baseline

def core_calypso(name, seed, s):
    from scripts.pso.calypso import CALYPSO
    p = problem(name)
    c = CALYPSO(exp="x", run=s, seed=seed, loss=p, lim=(p.lower, p.upper), name=name, N=N_S, D=N_X,
                maxIterCal=T, baseline="constant", velocityClamping=True, Ne=(30, 2), seterr="ignore")
    with contextlib.redirect_stdout(io.StringIO()):
        _, f, _ = c.iterate()
    return list(map(float, c.data["globalBests"])) + [float(f)]

def core_edaptive(name, seed):
    cfg = SolverConfig(identifier="core", fixed_CPs={"n_s": N_S, "w": W, "c_1": C, "c_2": C, "V_max_fraction": 1.0})
    _, h = BasicSolver(PSO, None, problem=problem(name), config=cfg, rnd_seed=seed).run(T)
    return list(map(float, h["f_best"]))

def task(t):
    kind, side, name, seed, s = t
    if kind == "bs":
        traj = bs_calypso(name, seed, s) if side == "calypso" else bs_edaptive(name, seed)
    else:
        traj = core_calypso(name, seed, s) if side == "calypso" else core_edaptive(name, seed)
    return (kind, side, name, s), [traj[k] for k in CHECKPOINTS]

def problem_checks():
    from scripts.pso.calypso import CALYPSO
    p = BBOB_FID17_IID2(N_X)
    rng = np.random.default_rng(0)
    inside = rng.uniform(p.lower, p.upper, (200, N_X))
    outside = inside * 3.0  # most rows outside [-5, 5]
    X = np.vstack([inside, outside])
    batch = p(X)
    rows = np.array([p(x) for x in X])
    c = CALYPSO(exp="x", run=0, seed=1, loss=p, lim=(p.lower, p.upper), name="b", N=N_S, D=N_X,
                maxIterCal=1, baseline="constant", velocityClamping=True, Ne=(30, 2), seterr="ignore")
    cal = c.getLoss(X)
    feasible = np.all((X >= p.lower) & (X <= p.upper), axis=1)
    before = p(inside)
    p(rng.uniform(-50, 50, (5000, N_X)))  # many infeasible evaluations
    after = p(inside)
    fresh = BBOB_FID17_IID2(N_X)(inside)
    return {
        "bounds": [float(p.lower.min()), float(p.lower.max()), float(p.upper.min()), float(p.upper.max())],
        "calypso_lim_is_problem_bounds": bool(np.array_equal(c.low, p.lower) and np.array_equal(c.high, p.upper)),
        "batch_equals_rows": bool(np.array_equal(batch, rows)),
        "calypso_getLoss_equals_problem_on_feasible": bool(np.array_equal(cal[feasible], batch[feasible])),
        "calypso_getLoss_inf_on_infeasible": bool(np.all(np.isinf(cal[~feasible]))),
        "n_feasible_of_400": int(feasible.sum()),
        "no_state_after_infeasible_evaluations": bool(np.array_equal(before, after)),
        "fresh_instance_equal": bool(np.array_equal(before, fresh)),
        "f_range_inside": [float(before.min()), float(before.max())],
    }

def main():
    out = {"problem_checks": problem_checks()}
    print(json.dumps(out["problem_checks"], indent=1), flush=True)
    tasks = []
    for name in FUNCS:
        for b in (6_000_000, 7_000_000):
            for s in range(30):
                seed = trial_seed(name, N_X, s, base=b)
                idx = (b // 1_000_000) * 100 + s
                for kind in (("bs", "core") if name.startswith("BBOB") else ("core",)):
                    for side in ("calypso", "edaptive"):
                        tasks.append((kind, side, name, seed, idx))
    tasks.sort(key=lambda t: t[1] != "calypso")
    res = {}
    with Pool(4, maxtasksperchild=40) as pool:
        for n, (k, v) in enumerate(pool.imap_unordered(task, tasks), 1):
            res[k] = v
            if n % 40 == 0:
                print(f"{n}/{len(tasks)}", flush=True)
    table = {}
    for kind, name in (("bs", "BBOB_FID17_IID2"), ("core", "BBOB_FID17_IID2"), ("core", "Rosenbrock")):
        c = np.array([v for (k, sd, nm, _), v in res.items() if (k, sd, nm) == (kind, "calypso", name)])
        e = np.array([v for (k, sd, nm, _), v in res.items() if (k, sd, nm) == (kind, "edaptive", name)])
        rows = []
        for j, cp in enumerate(CHECKPOINTS):
            p = mannwhitneyu(c[:, j], e[:, j]).pvalue if not np.all(np.r_[c[:, j], e[:, j]] == c[0, j]) else 1.0
            rows.append({"iteration": cp, "median_calypso": float(np.median(c[:, j])),
                         "median_edaptive": float(np.median(e[:, j])), "p": float(p)})
        examples = {str(s): {"calypso": res[(kind, "calypso", name, s)], "edaptive": res[(kind, "edaptive", name, s)]}
                    for s in (600, 601, 602)}
        table[f"{kind}/{name}"] = {"n_runs_per_side": len(c), "checkpoints": rows, "examples_seed_index": examples}
    out["trajectories"] = table
    out["checkpoints"] = CHECKPOINTS
    path = REPO / "results/bs_sapso_equivalence_seed7000000/bbob_investigation.json"
    path.write_text(json.dumps(out, indent=1) + "\n")
    for key, t in table.items():
        print("\n" + key, f"({t['n_runs_per_side']} runs per side; median gbest calypso / edaptive, MWU p)")
        for r in t["checkpoints"]:
            print(f"  iter {r['iteration']:5d}: {r['median_calypso']:12.4f} {r['median_edaptive']:12.4f}  p={r['p']:.3f}")
    print("wrote", path)

if __name__ == "__main__":
    main()
