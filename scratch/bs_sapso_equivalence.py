"""BS-SAPSO equivalence check: calypso's CALYPSO vs edaptive's BSSAPSO.

Run from the repo root (not a timed run; pinning to the P-cores is only for speed):

    nohup taskset -c 0,2,4,6 python scratch/bs_sapso_equivalence.py --workers 4 \\
        > results/bs_sapso_equivalence/run.log 2>&1 &

Configurations, in edaptive's terms (calypso's ``--vc --tc fixed <p> --sc <s> 30``
with ``p = update_freq - 1``, and ``s`` = ``improve`` for ``"improvement"``):
improvement and elitist at ``update_freq = 201`` (``p = 200``; improvement is the
published ``vc_fixed200_improve30``) and at ``update_freq = 21`` (``p = 20``).
edaptive's side uses ``experiments/configs/bs_sapso.json`` with ``update_freq``
and ``selection`` filled in.

calypso is run by building ``CALYPSO`` directly (not through ``execute.py`` or
``BS_PSO``, which use one seed list for every function and delete calypso's
output directories). Its keyword arguments come from ``execute.py``'s own code
(``calypso_kwargs``): the side-effect-free top of the script (imports, argument
parser, belief space, trigger and selection, experiment name) is executed with
the configuration's argv, then every keyword expression of its ``CALYPSO(...)``
call is evaluated, except ``savePath`` (``False``, nothing is written) and
``seed``. ``check_execute_kwargs`` also asserts the result against hand-written
values for ``vc_fixed200_improve30``. The problem is passed as ``execute.py``
does for ``--problems``: ``loss`` = the instance, ``lim = (lower, upper)``,
``name`` = the class name.

Seeds: run ``s`` of a function uses ``trial_seed(name, n_x, s, base=seed_base)``
on both sides (calypso: ``np.random.seed``; edaptive: ``rnd_seed``).

Everything is reported in edaptive's step ``t`` (calypso's iteration
``i = t + 1``). Per run, the worker returns: the final gbest; the steps whose box
differs from the previous step's (an update after step ``t - 1``) and the box in
force at steps 250, 1000, 2500 and 4999 (calypso: ``data['wmin']`` ...
``data['c2max']``, stored at the start of each iteration; edaptive: derived from
``history['belief_space']``, see ``run_edaptive``); the collapse step (calypso's
``bsConverged - 1``); and per update the step, the number of particles selected
and the number of feasible particles (calypso: by wrapping the instance's
``setBeliefSpace``, which reads ``trigger``, ``Ne`` and ``L[-1]`` before calling
the original, with no RNG and no behaviour change; edaptive: derived from the
history, ``n_feasible`` as recorded). So the exact checks compare calypso's
observed schedule, box changes and selection sizes with the rules BSSAPSO
documents, applied to its recorded history.

Exact checks (must agree in every run, up to the earlier collapse of the two runs
of a seed): the update steps, the box-change steps and the number of particles
selected at each update. Statistical checks: two-sided Mann-Whitney U across
seeds, with both medians, on the final gbest, on each CP's lower and upper bound
at the steps above, and on the collapse step (never = ``max_iter``) where any run
collapses. Holm over all tests together. Also the share of raw p < 0.05 and a KS
test of the raw p-values against uniform (the tests are not independent: the
bounds of one run are correlated, and the configurations share their seeds). For
elitist, the share of updates with fewer feasible particles than ``n_e``.

No bit-for-bit agreement is expected: calypso draws from numpy's global RNG.
Report the result; do not tune anything to make them agree.
"""

import os

for _variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_variable, "1")

import argparse  # noqa: E402
import ast  # noqa: E402
import contextlib  # noqa: E402
import io  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from multiprocessing import Pool  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
from scipy.stats import kstest, mannwhitneyu  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
CALYPSO_ROOT = REPO / "calypso"
EXECUTE_PY = CALYPSO_ROOT / "scripts" / "pso" / "execute.py"
CONFIG_PATH = REPO / "experiments" / "configs" / "bs_sapso.json"
sys.path.insert(0, str(CALYPSO_ROOT))

from edaptive.problems.ela_benchmark_suite import (  # noqa: E402
    BBOB_FID17_IID2,
    Brown,
    GeneralizedEggCrate,
    Qing,
    Rosenbrock,
    Step3,
)
from edaptive.solvers.pso.bs_sapso import BSSAPSO  # noqa: E402
from edaptive.utils.experiment_utils import parse_config  # noqa: E402
from edaptive.utils.seeding import trial_seed  # noqa: E402

FUNCTIONS = {
    cls.__name__: cls
    for cls in (Rosenbrock, Qing, GeneralizedEggCrate, Brown, BBOB_FID17_IID2, Step3)
}
# (selection, update_freq) in edaptive's terms.
CONFIGURATIONS = (
    ("improvement", 201),
    ("elitist", 201),
    ("improvement", 21),
    ("elitist", 21),
)
CALYPSO_SELECTIONS = {"improvement": "improve", "elitist": "elitist"}
N_E = 30
CP_NAMES = ("w", "c_1", "c_2")
CALYPSO_BOX_KEYS = (("wmin", "wmax"), ("c1min", "c1max"), ("c2min", "c2max"))
STEPS = (250, 1000, 2500, 4999)
DEFAULT_SEED_BASE = 6_000_000


def config_name(selection, update_freq):
    return f"{selection}_update_freq{update_freq}"


# --------------------------------------------------------------- calypso side


def execute_argv(calypso_selection, p, n_x, n_s, max_iter):
    return [
        "--vc",
        "--tc", "fixed", str(p),
        "--sc", calypso_selection, str(N_E),
        "--dimensions", str(n_x),
        "--N", str(n_s),
        "--max-iter-cal", str(max_iter),
    ]  # fmt: skip


def calypso_kwargs(argv, run, loss, lim, name):
    """``CALYPSO`` keyword arguments as ``execute.py`` builds them for ``argv``,
    without ``savePath`` and ``seed``, and the experiment name ``exp``."""
    tree = ast.parse(EXECUTE_PY.read_text())

    prefix = []
    for node in tree.body:
        targets = node.targets if isinstance(node, ast.Assign) else []
        names = {t.id for t in targets if isinstance(t, ast.Name)}
        if "expDIR" in names:
            break  # file management and runs follow
        if "args" in names:
            # args = parser.parse_args() -> parser.parse_args(ARGV)
            node.value.args = [
                ast.copy_location(ast.Name("ARGV", ast.Load()), node.value)
            ]
        prefix.append(node)
    namespace = {"ARGV": argv, "__name__": "execute_prefix"}
    exec(compile(ast.Module(prefix, []), str(EXECUTE_PY), "exec"), namespace)

    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "CALYPSO"
    ]
    assert len(calls) == 1, "expected one CALYPSO(...) call in execute.py"
    local = {"run": run, "loss": loss, "lim": lim, "name": name}
    kwargs = {}
    for keyword in calls[0].keywords:
        if keyword.arg in ("savePath", "seed"):
            continue
        expression = ast.Expression(keyword.value)
        kwargs[keyword.arg] = eval(
            compile(expression, str(EXECUTE_PY), "eval"), namespace, local
        )
    return kwargs, namespace["exp"]


def check_execute_kwargs(n_x, n_s, max_iter):
    """Assert execute.py's arguments for vc_fixed200_improve30 against the
    expected values. Returns them (as strings) for the report."""
    import pandas as pd

    problem = Rosenbrock(n_x)
    lim = (problem.lower, problem.upper)
    kwargs, exp = calypso_kwargs(
        execute_argv("improve", 200, n_x, n_s, max_iter), 0, problem, lim, "Rosenbrock"
    )
    expected_space = pd.DataFrame(
        {
            "w": {"min": 0, "max": 1},
            "c1": {"min": 0, "max": 4},
            "c2": {"min": 0, "max": 4},
        }
    )
    expected = {
        "exp": "vc_fixed200_improve30",
        "run": 0,
        "N": n_s,
        "D": n_x,
        "maxIterCal": max_iter,
        "baseline": None,
        "imitate": False,
        "velocityClamping": True,
        "bsStabilityGuided": False,
        "bsTriggerCriteria": "fixed",
        "bsPatience": 200,
        "bsSelectionCriteria": "improve",
        "Ne": (30, 2),
        "seterr": "ignore",
        "name": "Rosenbrock",
    }
    assert exp == expected["exp"], exp
    assert kwargs["exp"] == expected["exp"]
    assert kwargs["loss"] is problem and kwargs["lim"] is lim
    assert kwargs["beliefSpace"].equals(expected_space), kwargs["beliefSpace"]
    for key, value in expected.items():
        assert kwargs[key] == value, (key, kwargs[key], value)
    assert set(kwargs) == set(expected) | {"loss", "lim", "beliefSpace"}, set(kwargs)
    return {"exp": exp} | {
        key: repr(value)
        for key, value in kwargs.items()
        if key not in ("loss", "lim", "beliefSpace")
    } | {"beliefSpace": kwargs["beliefSpace"].to_dict()}


# Problem instances per worker process (BBOB construction is slow).
_PROBLEMS = {}


def problem_instance(function, n_x):
    if (function, n_x) not in _PROBLEMS:
        _PROBLEMS[(function, n_x)] = FUNCTIONS[function](n_x)
    return _PROBLEMS[(function, n_x)]


def summarise(f_final, lbs, ubs, collapse_step, updates, max_iter):
    """The per-run quantities the analysis needs. ``lbs``/``ubs``: the box in
    force at every step, shape ``(max_iter, n_CPs)``."""
    box = np.hstack([lbs, ubs])
    changes = np.flatnonzero(np.any(box[1:] != box[:-1], axis=1)) + 1
    steps = [t for t in STEPS if t < max_iter]
    return {
        "f_final": float(f_final),
        "box_change_steps": changes.tolist(),
        "lbs_at_steps": lbs[steps].tolist(),
        "ubs_at_steps": ubs[steps].tolist(),
        "collapse_step": collapse_step,
        "updates": updates,  # (t, n_e, n_feasible)
    }


def run_calypso(selection, update_freq, function, seed, run, n_x, n_s, max_iter):
    from scripts.pso.calypso import CALYPSO

    problem = problem_instance(function, n_x)
    lim = (problem.lower, problem.upper)
    argv = execute_argv(
        CALYPSO_SELECTIONS[selection], update_freq - 1, n_x, n_s, max_iter
    )
    kwargs, _ = calypso_kwargs(argv, run, problem, lim, function)
    calypso = CALYPSO(**kwargs, seed=seed, savePath=False)

    updates = []
    original = calypso.setBeliefSpace

    def recording(i):
        if calypso.trigger is True:
            n_feasible = int(np.isfinite(calypso.L[-1]).sum())
            updates.append((i - 1, min(calypso.Ne, n_s), n_feasible))
        original(i)

    calypso.setBeliefSpace = recording

    with contextlib.redirect_stdout(io.StringIO()):  # pl() on convergence
        _, f_final, _ = calypso.iterate()

    lbs = np.column_stack([calypso.data[lo] for lo, _ in CALYPSO_BOX_KEYS])
    ubs = np.column_stack([calypso.data[hi] for _, hi in CALYPSO_BOX_KEYS])
    converged = calypso.bsConverged  # calypso's iteration i, or False
    collapse_step = None if converged is False else int(converged) - 1
    return summarise(
        f_final, lbs.astype(float), ubs.astype(float), collapse_step, updates, max_iter
    )


# -------------------------------------------------------------- edaptive side


def edaptive_config(selection, update_freq, n_s):
    solver_config, _ = parse_config(json.loads(CONFIG_PATH.read_text()))
    assert solver_config.fixed_CPs["n_s"] == n_s
    assert solver_config.adapter_params["n_e"] == N_E
    return solver_config.with_values(
        {"update_freq": update_freq, "selection": selection}
    )


def run_edaptive(selection, update_freq, function, seed, n_x, n_s, max_iter):
    """``BSSAPSO.run``; everything else is derived from ``history["belief_space"]``:
    update ``k`` (1-based) comes after step ``k * update_freq - 1``, the box in force
    at step ``t`` is that of the last update ``k <= t // update_freq`` (the CP
    bounds before the first), update ``k`` selects ``max(n_e - (k - 1), n_e_min)``
    particles for elitist and ``n_e`` for improvement, and a run with fewer
    updates than its schedule has collapsed, detected after step
    ``n_updates * update_freq``."""
    config = edaptive_config(selection, update_freq, n_s)
    solver = BSSAPSO(
        problem=problem_instance(function, n_x), config=config, rnd_seed=seed
    )
    f_final, history = solver.run(max_iter)
    belief_space = history["belief_space"]
    n_updates = len(belief_space["n_feasible"])
    space = config.adapted_CP_space
    n_e, n_e_min = config.adapter_params["n_e"], config.adapter_params["n_e_min"]

    # Box per step: row 0 = the CP bounds, row k = after update k.
    box_lbs = np.vstack([space.lower_bounds, belief_space["belief_lbs"]])
    box_ubs = np.vstack([space.upper_bounds, belief_space["belief_ubs"]])
    k_at_step = np.minimum(np.arange(max_iter) // update_freq, n_updates)
    lbs, ubs = box_lbs[k_at_step], box_ubs[k_at_step]

    updates = []
    for k in range(1, n_updates + 1):
        n_selected = max(n_e - (k - 1), n_e_min) if selection == "elitist" else n_e
        n_feasible = int(belief_space["n_feasible"][k - 1])
        updates.append((k * update_freq - 1, min(n_selected, n_s), n_feasible))
    collapsed = n_updates < max_iter // update_freq
    collapse_step = n_updates * update_freq if collapsed else None
    return summarise(f_final, lbs, ubs, collapse_step, updates, max_iter)


def run_task(task):
    side, selection, update_freq, function, s, seed, n_x, n_s, max_iter = task
    if side == "calypso":
        result = run_calypso(
            selection, update_freq, function, seed, s, n_x, n_s, max_iter
        )
    else:
        result = run_edaptive(
            selection, update_freq, function, seed, n_x, n_s, max_iter
        )
    return (side, config_name(selection, update_freq), function, s), result


# ------------------------------------------------------------------- analysis


def exact_checks(runs_calypso, runs_edaptive, max_iter):
    """Per seed: update steps, box-change steps and particles selected per update
    agree up to the earlier collapse (or the end of the run)."""
    mismatches = []
    for s, (c, e) in enumerate(zip(runs_calypso, runs_edaptive, strict=True)):
        horizon = min(
            max_iter if c["collapse_step"] is None else c["collapse_step"],
            max_iter if e["collapse_step"] is None else e["collapse_step"],
        )
        # An update after step t shows as a box change at step t + 1.
        checked = {}
        for side, run in (("calypso", c), ("edaptive", e)):
            checked[side] = {
                "update_steps": [t for t, _, _ in run["updates"] if t < horizon],
                "n_e": [n for t, n, _ in run["updates"] if t < horizon],
                "box_change_steps": [
                    t for t in run["box_change_steps"] if t <= horizon
                ],
            }
        if checked["calypso"] != checked["edaptive"]:
            mismatches.append({"seed_index": s, "horizon": horizon} | checked)
    return {
        "n_runs": len(runs_calypso),
        "n_match": len(runs_calypso) - len(mismatches),
        "mismatches": mismatches,
    }


def mwu(x_calypso, x_edaptive):
    x_calypso, x_edaptive = np.asarray(x_calypso), np.asarray(x_edaptive)
    if np.all(np.concatenate([x_calypso, x_edaptive]) == x_calypso[0]):
        p = 1.0  # one value on both sides; mannwhitneyu is undefined
    else:
        p = float(mannwhitneyu(x_calypso, x_edaptive).pvalue)
    return {
        "median_calypso": float(np.median(x_calypso)),
        "median_edaptive": float(np.median(x_edaptive)),
        "p": p,
    }


def statistical_tests(runs_calypso, runs_edaptive, max_iter):
    tests = {
        "f_final": mwu(
            [r["f_final"] for r in runs_calypso], [r["f_final"] for r in runs_edaptive]
        )
    }
    steps = [t for t in STEPS if t < max_iter]
    for k, name in enumerate(CP_NAMES):
        for bound, key in (("lb", "lbs_at_steps"), ("ub", "ubs_at_steps")):
            for j, t in enumerate(steps):
                tests[f"{name}.{bound}.step_{t}"] = mwu(
                    [r[key][j][k] for r in runs_calypso],
                    [r[key][j][k] for r in runs_edaptive],
                )

    def collapse(run):
        return max_iter if run["collapse_step"] is None else run["collapse_step"]

    collapse_c = [collapse(r) for r in runs_calypso]
    collapse_e = [collapse(r) for r in runs_edaptive]
    if min(collapse_c + collapse_e) < max_iter:
        tests["collapse_step"] = mwu(collapse_c, collapse_e)
    return tests


def feasibility_shortfall(runs):
    """Share of updates with fewer feasible particles than particles selected."""
    updates = [(n, n_f) for r in runs for _, n, n_f in r["updates"]]
    if not updates:
        return {"n_updates": 0, "share_n_feasible_below_n_e": None}
    short = sum(n_f < n for n, n_f in updates)
    return {
        "n_updates": len(updates),
        "share_n_feasible_below_n_e": short / len(updates),
        "median_n_feasible": float(np.median([n_f for _, n_f in updates])),
    }


def holm(p_values):
    """Holm step-down adjusted p-values."""
    p = np.asarray(p_values, dtype=float)
    adjusted = np.empty_like(p)
    running = 0.0
    for i, index in enumerate(np.argsort(p, kind="stable")):
        running = max(running, min(1.0, (len(p) - i) * p[index]))
        adjusted[index] = running
    return adjusted


# ----------------------------------------------------------------------- main


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seeds", type=int, default=30)
    parser.add_argument("--seed-base", type=int, default=DEFAULT_SEED_BASE)
    parser.add_argument("--n-x", type=int, default=30)
    parser.add_argument("--n-s", type=int, default=30)
    parser.add_argument("--max-iter", type=int, default=5000)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--out", type=Path, default=REPO / "results" / "bs_sapso_equivalence"
    )
    args = parser.parse_args()

    started = time.perf_counter()
    execute_check = check_execute_kwargs(args.n_x, args.n_s, args.max_iter)

    seeds = {
        f: [trial_seed(f, args.n_x, s, base=args.seed_base) for s in range(args.seeds)]
        for f in FUNCTIONS
    }
    tasks = [
        (side, sel, uf, f, s, seeds[f][s], args.n_x, args.n_s, args.max_iter)
        for f in FUNCTIONS
        for sel, uf in CONFIGURATIONS
        for s in range(args.seeds)
        for side in ("calypso", "edaptive")
    ]
    # Slowest first (BBOB, calypso), so the pool ends evenly.
    tasks.sort(key=lambda task: (task[3] != "BBOB_FID17_IID2", task[0] != "calypso"))

    results = {}
    with Pool(args.workers, maxtasksperchild=50) as pool:
        for n_done, (key, result) in enumerate(
            pool.imap_unordered(run_task, tasks), start=1
        ):
            results[key] = result
            if n_done % 20 == 0 or n_done == len(tasks):
                elapsed = time.perf_counter() - started
                print(f"{n_done}/{len(tasks)} runs, {elapsed:.0f} s", flush=True)

    report = {}
    for selection, update_freq in CONFIGURATIONS:
        name = config_name(selection, update_freq)
        report[name] = {}
        for f in FUNCTIONS:
            runs_c = [results[("calypso", name, f, s)] for s in range(args.seeds)]
            runs_e = [results[("edaptive", name, f, s)] for s in range(args.seeds)]
            entry = {
                "exact": exact_checks(runs_c, runs_e, args.max_iter),
                "tests": statistical_tests(runs_c, runs_e, args.max_iter),
                "n_collapsed": {
                    "calypso": sum(r["collapse_step"] is not None for r in runs_c),
                    "edaptive": sum(r["collapse_step"] is not None for r in runs_e),
                },
            }
            if selection == "elitist":
                entry["feasibility_shortfall"] = {
                    "calypso": feasibility_shortfall(runs_c),
                    "edaptive": feasibility_shortfall(runs_e),
                }
            report[name][f] = entry

    keys = [(c, f, q) for c in report for f in report[c] for q in report[c][f]["tests"]]
    p_raw = np.array([report[c][f]["tests"][q]["p"] for c, f, q in keys])
    for (c, f, q), p_holm in zip(keys, holm(p_raw), strict=True):
        report[c][f]["tests"][q]["p_holm"] = float(p_holm)
    ks = kstest(p_raw, "uniform")
    n_exact = sum(report[c][f]["exact"]["n_match"] for c in report for f in report[c])
    n_runs = sum(report[c][f]["exact"]["n_runs"] for c in report for f in report[c])
    summary = {
        "exact_matches": f"{n_exact}/{n_runs}",
        "n_tests": len(p_raw),
        "share_raw_p_below_0.05": float(np.mean(p_raw < 0.05)),
        "ks_raw_p_vs_uniform": {"statistic": float(ks.statistic), "p": float(ks.pvalue)},
        "min_p_raw": float(p_raw.min()),
        "min_p_holm": float(min(report[c][f]["tests"][q]["p_holm"] for c, f, q in keys)),
        "n_holm_below_0.05": int(
            sum(report[c][f]["tests"][q]["p_holm"] < 0.05 for c, f, q in keys)
        ),
    }
    output = {
        "settings": {
            "functions": list(FUNCTIONS),
            "configurations": {
                config_name(sel, uf): {
                    "selection": sel,
                    "update_freq": uf,
                    "calypso": f"--vc --tc fixed {uf - 1} "
                    f"--sc {CALYPSO_SELECTIONS[sel]} {N_E}",
                }
                for sel, uf in CONFIGURATIONS
            },
            "n_e": N_E,
            "seeds": args.seeds,
            "seed_base": args.seed_base,
            "n_x": args.n_x,
            "n_s": args.n_s,
            "max_iter": args.max_iter,
            "steps": [t for t in STEPS if t < args.max_iter],
            "edaptive_config": str(CONFIG_PATH.relative_to(REPO)),
        },
        "execute_kwargs_vc_fixed200_improve30": execute_check,
        "summary": summary,
        "results": report,
        "wall_seconds": time.perf_counter() - started,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / "equivalence.json"
    path.write_text(json.dumps(output, indent=1) + "\n")

    print(json.dumps(summary, indent=1))
    for c in report:
        for f in report[c]:
            entry = report[c][f]
            worst = min(entry["tests"].items(), key=lambda item: item[1]["p"])
            gbest = entry["tests"]["f_final"]
            line = (
                f"{c:26s} {f:20s} exact {entry['exact']['n_match']}/"
                f"{entry['exact']['n_runs']}  f_final median "
                f"{gbest['median_calypso']:.4g} / {gbest['median_edaptive']:.4g} "
                f"p={gbest['p']:.3f}  min p {worst[0]}={worst[1]['p']:.3g}"
            )
            if "feasibility_shortfall" in entry:
                fs = entry["feasibility_shortfall"]
                line += (
                    f"  n_feasible<n_e "
                    f"{fs['calypso']['share_n_feasible_below_n_e']:.2f}"
                    f"/{fs['edaptive']['share_n_feasible_below_n_e']:.2f}"
                )
            print(line)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
