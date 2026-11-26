"""SAC training's solver time in edaptive's cost model, next to calypso's raw solver time.

    python scratch/sac_equivalent_solver_time.py \
        --runs results/sac_training/vc_sac_j50/training_runs.csv \
        --timings results/experiment_results/sac_sapso_timing.pkl

`--runs` is the `training_runs.csv` of a SAC training (one row per calypso run: env,
run, function, seed, phase, steps, solver and problem-construction times), and
`--times` its `training_times.json` (default: the one next to `--runs`).
`--timings` is an `execute.py` results pkl of `SACSAPSO` with `--problem_set
training` (every TRAINING_SET function, n_x = 30, 5000 iterations, n_s = 30), run on
the same machine and thread settings as the training, with as many MPI ranks as the
I/F-race tunings whose solver time it is compared with (their tuning.log: "Number of
MPI processes"). Use the exported copy of training_runs.csv
(results/sac_training/<exp>/), since the training wrapper deletes calypso/data.

For every training run (phases initial_collect, collect and validation; reset_only
rows, resets that never ran a PSO iteration, are left out) the edaptive-equivalent
solver time is edaptive's mean per-run core time for that function: `core_time` of
`history["timings"]` without `_load_policy` (loading the policy file, which calypso
has no counterpart of). Initial-collect runs take random actions, but use the same
per-function figure.

Printed, and written to `equivalent_cost.json` next to `--runs`:
- per clock, the training's cost: calypso's solver time, the edaptive-equivalent solver
  time, configurator, total as measured (solver + configurator) and total with the
  edaptive-equivalent solver; setup, io, framework and problem_construction are listed
  separately, as excluded from total;
- per phase, calypso's and the equivalent solver time summed over the runs;
- per function, the mean per-run times and their ratio.

Caveats:
- calypso skips evaluating infeasible particles; edaptive evaluates every particle.
  The equivalent figure therefore uses edaptive's cost model and does not reproduce
  calypso's evaluation count.
- calypso's raw solver time also includes its own run-loop overhead (history lists,
  storeMetrics), the policy's action calls and the PSO runs' share of tf-agents'
  environment wrapper; edaptive's core time covers only its timed leaf methods.
- Problem construction is in neither figure (reported separately for calypso).
"""

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from edaptive.problems.ela_benchmark_suite import TRAINING_SET
from edaptive.utils.timing import CLOCKS

N_X, MAX_ITERATIONS, N_S = 30, 5000, 30
PHASES = ("initial_collect", "collect", "validation")
EXCLUDED_BUCKETS = ("setup", "io", "framework", "problem_construction")


def core_time_without_policy_load(timings):
    """`core_time` of one run's timings, leaving out `SACSAPSO._load_policy`."""
    totals = {clock: 0.0 for clock in CLOCKS}
    for component, methods in timings.items():
        for method, times in methods.items():
            if component == "solver" and method == "_load_policy":
                continue
            for clock in CLOCKS:
                totals[clock] += times[clock]
    return totals


def per_function_core_times(results):
    """Mean (and count) of the per-run core time of each function, per clock."""
    rows = {}
    for function, trials in results.items():
        for trial, history in trials.items():
            n_iterations = len(history["f_best"]) - 1
            n_s, n_x = history["velocities"][0].shape
            if (n_iterations, n_s, n_x) != (MAX_ITERATIONS, N_S, N_X):
                raise ValueError(
                    f"{function} trial {trial}: {n_iterations} iterations, n_s = {n_s}, "
                    f"n_x = {n_x}; expected {MAX_ITERATIONS}, {N_S}, {N_X}"
                )
        times = [core_time_without_policy_load(h["timings"]) for h in trials.values()]
        rows[function] = {
            "n_trials": len(times),
            **{
                f"edaptive_{clock}": np.mean([t[clock] for t in times])
                for clock in CLOCKS
            },
        }
    return pd.DataFrame.from_dict(rows, orient="index")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--runs", required=True, help="training_runs.csv")
    parser.add_argument("--timings", required=True, help="execute.py results pkl")
    parser.add_argument("--times", help="training_times.json (default: next to --runs)")
    args = parser.parse_args()
    runs_path = Path(args.runs)
    times_path = (
        Path(args.times) if args.times else runs_path.parent / "training_times.json"
    )
    with open(times_path) as file:
        training = json.load(file)["total"]

    runs = pd.read_csv(args.runs)
    n_reset_only = int((runs["phase"] == "reset_only").sum())
    runs = runs[runs["phase"].isin(PHASES)].copy()

    with open(args.timings, "rb") as file:
        edaptive = per_function_core_times(pickle.load(file))

    missing = sorted(set(runs["function"]) - set(edaptive.index))
    if missing:
        raise SystemExit(f"No edaptive timings for {missing}")
    not_training = sorted(set(runs["function"]) - {p.__name__ for p in TRAINING_SET})
    if not_training:
        print(f"Note: functions outside TRAINING_SET: {not_training}")

    for clock in CLOCKS:
        runs[f"edaptive_{clock}"] = runs["function"].map(edaptive[f"edaptive_{clock}"])

    columns = {
        "runs": ("function", "size"),
        **{
            f"{name}_{clock}": (f"{column}_{clock}", "sum")
            for clock in CLOCKS
            for name, column in (
                ("calypso_solver", "solver"),
                ("edaptive_equivalent", "edaptive"),
                ("calypso_problem_construction", "problem_construction"),
            )
        },
    }
    by_phase = runs.groupby("phase").agg(**columns).reindex(PHASES)
    by_phase.loc["all"] = by_phase.sum()
    for clock in CLOCKS:
        by_phase[f"ratio_{clock}"] = (
            by_phase[f"calypso_solver_{clock}"]
            / by_phase[f"edaptive_equivalent_{clock}"]
        )

    by_function = runs.groupby("function").agg(
        runs=("function", "size"),
        **{
            f"calypso_solver_{clock}_mean": (f"solver_{clock}", "mean")
            for clock in CLOCKS
        },
    )
    by_function = by_function.join(edaptive)
    for clock in CLOCKS:
        by_function[f"ratio_{clock}"] = (
            by_function[f"calypso_solver_{clock}_mean"]
            / by_function[f"edaptive_{clock}"]
        )

    # The training's cost per clock. calypso's solver and the configurator are the
    # training_times.json buckets; the equivalent solver is the sum over the runs.
    equivalent = by_phase.loc["all"]
    combined = pd.DataFrame(
        {
            clock: {
                "calypso_solver": training["solver"][clock],
                "edaptive_equivalent_solver": equivalent[
                    f"edaptive_equivalent_{clock}"
                ],
                "configurator": training["configurator"][clock],
                "total_measured": training["total"][clock],
                "total_equivalent": equivalent[f"edaptive_equivalent_{clock}"]
                + training["configurator"][clock],
            }
            for clock in CLOCKS
        }
    )
    excluded = pd.DataFrame(
        {clock: {b: training[b][clock] for b in EXCLUDED_BUCKETS} for clock in CLOCKS}
    )

    pd.set_option("display.width", 200)
    print(f"Runs compared: {len(runs)} ({n_reset_only} reset_only rows left out)\n")
    print("Training cost (seconds; total = solver + configurator):")
    print(combined.round(2).to_string())
    print("\nExcluded from total (seconds):")
    print(excluded.round(2).to_string())
    print("\nSeconds, summed over the training runs, per phase:")
    print(by_phase.round(2).to_string())
    print("\nPer function (seconds per run):")
    print(by_function.round(4).to_string())

    output = runs_path.parent / "equivalent_cost.json"
    result = {
        "runs": str(runs_path),
        "times": str(times_path),
        "timings": args.timings,
        "n_runs": len(runs),
        "n_reset_only": n_reset_only,
        "combined": combined.to_dict(),
        "excluded": excluded.to_dict(),
        "by_phase": by_phase.to_dict(orient="index"),
        "by_function": by_function.to_dict(orient="index"),
    }
    with open(output, "w") as file:
        json.dump(result, file, indent=2, default=float)
    print(f"\nWritten to {output}")


if __name__ == "__main__":
    main()
