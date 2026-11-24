"""SAC trainer comparison: calypso's original (Reverb) trainer vs the det trainer.

    python scratch/sac_trainer_comparison.py determinism [--parallel 3]
    python scratch/sac_trainer_comparison.py train [--parallel 10]
    python scratch/sac_trainer_comparison.py evaluate [--trials 30] [--processes 16]

determinism: two smoke trainings with --trainer det and the same seed (plus one
  with --trainer original, to check it still runs); compares validation_results.csv,
  selected_policy.json and the variables of every validated policy.
train: the reduced-scale comparison, 5 training seeds per trainer: 6 training
  functions (one BBOB), 30-D, N = 30, 1000 iterations, budget 480 runs; the other
  settings as experiments/scripts/train_sac.py.
  Also repeats the first seed once per trainer (REPEAT_RUNS).
evaluate: exports each selected policy (calypso's exportPolicy, with its checks),
  runs it in edaptive's SACSAPSO on fresh seeds of the same training functions
  (trial_seed with base FRESH_SEED_BASE, unrelated to the training and validation
  seeds), and compares the trainers with two-sided Mann-Whitney U tests, the unit
  being the training seed (5 vs 5):
  - validation, per checkpoint: each training's mean over functions of the rank of
    its f_final among the 10 trainings (for that function and checkpoint);
  - fresh seeds, overall: each policy's mean rank over all (function, trial) pairs,
    ranked among the 10 policies (the same seed for every policy);
  - fresh seeds, per function: each policy's median final gbest; Holm over functions.
  With 5 vs 5 the smallest two-sided p is 2/252 = 0.0079, so this only catches
  large differences.

Trainings run calypsoTrainSAC directly (cwd calypso/), each with its own experiment
name (--text), so calypso/data is not deleted. All runs of one command start
together: each reads misc/pkl/funcRanges.pkl at start and rewrites it on exit
(logging only; rewards do not use it). Results go to results/sac_trainer_comparison/.
"""

import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, rankdata

from edaptive.problems import ela_benchmark_suite as suite

REPO = Path(__file__).resolve().parents[1]
CALYPSO = REPO / "calypso"
OUT = REPO / "results" / "sac_trainer_comparison"

TRAINERS = ("original", "det")
TRAINING_SEEDS = (101, 102, 103, 104, 105)
FRESH_SEED_BASE = 7_000_000

SMOKE_PROBLEMS = (suite.Rosenbrock, suite.Ripple25, suite.BBOB_FID17_IID2)
PROBLEMS = (
    suite.Schwefel1,
    suite.Ripple25,
    suite.Rosenbrock,
    suite.GeneralizedEggCrate,
    suite.Discus,
    suite.BBOB_FID17_IID2,
)

COMMON = {
    "j": 50, "gamma": 0.99, "rsf": 1, "ilr": 0.0001, "dcr": 1, "dcs": 100, "psi": 10,
    "nef": 1, "eval-seed-offset": 1_000_000,
}
SMOKE = COMMON | {
    "dimensions": 5, "N": 10, "max-iter-cal": 200, "i": 1000, "n": 32, "ics": 1,
    "ei": 1, "psi": 1, "budget-runs": 18,
}
REDUCED = COMMON | {
    "dimensions": 30, "N": 30, "max-iter-cal": 1000, "i": 200000, "n": 256, "ics": 5,
    "ei": 10, "budget-runs": 480,
}


def dotted(problem):
    return f"{problem.__module__}.{problem.__qualname__}"


def write_func_ranges(problems):
    """calypso's env needs every function in funcRanges.pkl; placeholder values as
    in CalypsoWrapper (they only affect the GBN printouts)."""
    pd.DataFrame(
        {"low": np.inf, "high": -np.inf, "delta": np.nan},
        index=[p.__name__ for p in problems],
    ).to_pickle(CALYPSO / "misc" / "pkl" / "funcRanges.pkl")


def command(settings, problems, trainer, seed, text):
    cmd = [sys.executable, "-m", "scripts.rl.calypsoTrainSAC"]
    for name, value in (settings | {"seed": seed}).items():
        cmd += [f"--{name}", str(value)]
    cmd += ["--train-problems", *map(dotted, problems)]
    cmd += ["--eval-problems", *map(dotted, problems)]
    cmd += ["--vc", "--trainer", trainer, "--text", text]
    return cmd


def exp_path(text):
    paths = glob.glob(str(CALYPSO / "data" / f"*_{text}"))
    if len(paths) != 1:
        raise FileNotFoundError(f"Expected one experiment *_{text}, found {paths}")
    return Path(paths[0])


def train_all(runs, settings, problems, parallel):
    """runs: [(trainer, seed, text)]; at most `parallel` at once."""
    for _, _, text in runs:
        for path in glob.glob(str(CALYPSO / "data" / f"*_{text}")):
            shutil.rmtree(path)
    write_func_ranges(problems)
    OUT.mkdir(parents=True, exist_ok=True)
    pending, running, times = list(runs), [], {}
    while pending or running:
        while pending and len(running) < parallel:
            trainer, seed, text = pending.pop(0)
            log = open(OUT / f"{text}.stdout", "w")
            proc = subprocess.Popen(
                command(settings, problems, trainer, seed, text),
                cwd=CALYPSO, stdout=log, stderr=subprocess.STDOUT,
            )
            running.append((proc, log, text, time.time()))
        time.sleep(5)
        for entry in list(running):
            proc, log, text, start = entry
            if proc.poll() is not None:
                log.close()
                running.remove(entry)
                times[text] = time.time() - start
                if proc.returncode:
                    raise RuntimeError(f"{text} failed; see {OUT / (text + '.stdout')}")
                print(f"{text} done in {times[text] / 60:.1f} min", flush=True)
    return times


def policy_variables(policy_dir):
    import tensorflow as tf

    saved = tf.saved_model.load(str(policy_dir))
    return {v.name: v.numpy() for v in saved.model_variables}


def determinism(args):
    runs = [
        ("det", 123, "smoke_det_a"),
        ("det", 123, "smoke_det_b"),
        ("original", 123, "smoke_original"),
    ]
    times = train_all(runs, SMOKE, SMOKE_PROBLEMS, args.parallel)
    a, b = exp_path("smoke_det_a"), exp_path("smoke_det_b")
    csv_a = (a / "validation_results.csv").read_text()
    csv_b = (b / "validation_results.csv").read_text()
    selected_a = json.loads((a / "selected_policy.json").read_text())
    selected_b = json.loads((b / "selected_policy.json").read_text())
    steps = sorted(p.name for p in (a / "model/policies/validated").iterdir())
    identical_policies = {}
    for step in steps:
        va = policy_variables(a / "model/policies/validated" / step)
        vb = policy_variables(b / "model/policies/validated" / step)
        identical_policies[step] = va.keys() == vb.keys() and all(
            np.array_equal(va[k], vb[k]) for k in va
        )
    result = {
        "identical_validation_results_csv": csv_a == csv_b,
        "validation_rows": len(csv_a.splitlines()) - 1,
        "identical_selected_policy_json": {
            k: v for k, v in selected_a.items() if k != "path"
        } == {k: v for k, v in selected_b.items() if k != "path"},
        "selected_step": selected_a["step"],
        "identical_validated_policy_variables": identical_policies,
        "original_ran": (exp_path("smoke_original") / "selected_policy.json").is_file(),
        "wall_seconds": times,
    }
    print(json.dumps(result, indent=2))
    (OUT / "determinism.json").write_text(json.dumps(result, indent=2))


def comparison_runs():
    return [(t, s, f"cmp_{t}_s{s}") for t in TRAINERS for s in TRAINING_SEEDS]


# Repeats of the first seed: the det repeat must reproduce its run exactly (a
# determinism check at the reduced scale); the original's shows its run-to-run
# variation for the same seed.
REPEAT_RUNS = [(t, TRAINING_SEEDS[0], f"cmp_{t}_s{TRAINING_SEEDS[0]}_repeat") for t in TRAINERS]


def train(args):
    times = train_all(comparison_runs() + REPEAT_RUNS, REDUCED, PROBLEMS, args.parallel)
    (OUT / "training_wall_seconds.json").write_text(json.dumps(times, indent=2))


def repeat_check():
    result = {}
    for trainer, seed, text in REPEAT_RUNS:
        a = pd.read_csv(exp_path(f"cmp_{trainer}_s{seed}") / "validation_results.csv")
        b = pd.read_csv(exp_path(text) / "validation_results.csv")
        sa = json.loads((exp_path(f"cmp_{trainer}_s{seed}") / "selected_policy.json").read_text())
        sb = json.loads((exp_path(text) / "selected_policy.json").read_text())
        steps = sorted(p.name for p in (exp_path(text) / "model/policies/validated").iterdir())
        identical = {}
        for step in steps:
            va = policy_variables(exp_path(f"cmp_{trainer}_s{seed}") / "model/policies/validated" / step)
            vb = policy_variables(exp_path(text) / "model/policies/validated" / step)
            identical[step] = all(np.array_equal(va[k], vb[k]) for k in va)
        result[trainer] = {
            "identical_validation_results_csv": a.equals(b),
            "rows_with_different_f_final": int((a["f_final"] != b["f_final"]).sum()),
            "rows": len(a),
            "selected_steps": [sa["step"], sb["step"]],
            "identical_validated_policy_variables": identical,
        }
    return result


def export(text):
    out = OUT / text
    subprocess.run(
        [sys.executable, "-m", "scripts.rl.exportPolicy", "--exp", exp_path(text).name,
         "--out", str(out)],
        cwd=CALYPSO, check=True, stdout=subprocess.DEVNULL,
    )
    return out / "policy.npz"


def fresh_run(job):
    from edaptive.core.config import SolverConfig
    from edaptive.solvers.pso.sac_sapso import SACSAPSO
    from edaptive.utils.seeding import trial_seed

    policy_path, problem_name, trial = job
    problem = getattr(suite, problem_name)(dimensions=REDUCED["dimensions"])
    config = SolverConfig(
        identifier="sac_sapso_cmp",
        fixed_CPs={"n_s": REDUCED["N"], "V_max_fraction": 1.0},
        adapted_CP_space=None,
        adapter_params={
            "policy_path": str(policy_path), "nt": REDUCED["j"],
            "max_iterations": REDUCED["max-iter-cal"],
        },
    )
    seed = trial_seed(problem_name, REDUCED["dimensions"], trial, base=FRESH_SEED_BASE)
    solver = SACSAPSO(None, None, problem=problem, config=config, rnd_seed=seed)
    f_best, _ = solver.run(max_iterations=REDUCED["max-iter-cal"])
    return float(f_best)


def mwu(x, y):
    return {
        "median_original": float(np.median(x)), "median_det": float(np.median(y)),
        "p": float(mannwhitneyu(x, y, alternative="two-sided").pvalue),
    }


def holm(ps):
    order = np.argsort(ps)
    adjusted, running = np.empty(len(ps)), 0.0
    for i, k in enumerate(order):
        running = max(running, min(1.0, (len(ps) - i) * ps[k]))
        adjusted[k] = running
    return adjusted


def evaluate(args):
    runs = comparison_runs()
    texts = [text for _, _, text in runs]
    is_det = np.array([t == "det" for t, _, _ in runs])

    # Validation performance per checkpoint (index k: the k-th validation).
    validation = {
        text: pd.read_csv(exp_path(text) / "validation_results.csv") for text in texts
    }
    n_checkpoints = {text: df["step"].nunique() for text, df in validation.items()}
    selected = {
        text: json.loads((exp_path(text) / "selected_policy.json").read_text())
        for text in texts
    }
    per_checkpoint = []
    for k in range(min(n_checkpoints.values())):
        f_final = []  # [training, function]
        for text in texts:
            df = validation[text]
            step = sorted(df["step"].unique())[k]
            rows = df[df["step"] == step].set_index("function")
            f_final.append([rows.loc[p.__name__, "f_final"] for p in PROBLEMS])
        ranks = rankdata(np.array(f_final), axis=0).mean(axis=1)
        per_checkpoint.append({"checkpoint": k + 1} | mwu(ranks[~is_det], ranks[is_det]))
    selected_ranks = []
    for text in texts:
        steps = sorted(validation[text]["step"].unique())
        selected_ranks.append(steps.index(selected[text]["step"]) + 1)

    # Fresh seeds.
    policies = {text: export(text) for text in texts}
    jobs = [
        (policies[text], p.__name__, trial)
        for text in texts for p in PROBLEMS for trial in range(args.trials)
    ]
    with ProcessPoolExecutor(args.processes) as pool:
        results = list(pool.map(fresh_run, jobs, chunksize=4))
    f = np.array(results).reshape(len(texts), len(PROBLEMS), args.trials)
    overall = rankdata(f, axis=0).mean(axis=(1, 2))
    per_function = [
        {"function": p.__name__} | mwu(
            np.median(f[~is_det, i], axis=1), np.median(f[is_det, i], axis=1)
        )
        for i, p in enumerate(PROBLEMS)
    ]
    for row, p_holm in zip(per_function, holm([r["p"] for r in per_function])):
        row["p_holm"] = float(p_holm)

    result = {
        "settings": REDUCED | {"functions": [p.__name__ for p in PROBLEMS],
                               "training_seeds": TRAINING_SEEDS,
                               "fresh_trials": args.trials,
                               "fresh_seed_base": FRESH_SEED_BASE},
        "note": "5 vs 5 training seeds: the smallest two-sided MWU p is 0.0079; "
                "only large differences can be detected.",
        "n_checkpoints": n_checkpoints,
        "selected_checkpoint": dict(zip(texts, selected_ranks)),
        "validation_mean_rank_per_checkpoint": per_checkpoint,
        "fresh_overall_mean_rank": {"per_policy": dict(zip(texts, overall.tolist()))}
        | mwu(overall[~is_det], overall[is_det]),
        "fresh_per_function_median": per_function,
        "fresh_per_policy_median": {
            text: {p.__name__: float(np.median(f[t, i])) for i, p in enumerate(PROBLEMS)}
            for t, text in enumerate(texts)
        },
    }
    result["same_seed_repeats"] = repeat_check()
    print(json.dumps(result, indent=2))
    (OUT / "comparison.json").write_text(json.dumps(result, indent=2))
    np.save(OUT / "fresh_f_final.npy", f)


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("determinism").add_argument("--parallel", type=int, default=3)
    sub.add_parser("train").add_argument("--parallel", type=int, default=12)
    p = sub.add_parser("evaluate")
    p.add_argument("--trials", type=int, default=30)
    p.add_argument("--processes", type=int, default=16)
    args = parser.parse_args()
    {"determinism": determinism, "train": train, "evaluate": evaluate}[args.command](args)


if __name__ == "__main__":
    main()
