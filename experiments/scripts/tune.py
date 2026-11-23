import os

# One BLAS/OpenMP thread per process, so each process's recorded times are its own
# compute effort (set before numpy is imported; a value from the shell wins).
os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse
import json
from pathlib import Path

from edaptive.problems.ela_benchmark_suite import TRAINING_SET
from edaptive.registry import resolve
from edaptive.tuning.ifrace import IFRace
from edaptive.utils.experiment_utils import parse_config


def parse_args():
    parser = argparse.ArgumentParser(description="")

    parser.add_argument("--solver", type=str, required=True, help="Solver to run.")

    parser.add_argument(
        "--optimizer", type=str, required=True, help="Optimizer to run."
    )
    parser.add_argument("--adapter", type=str, required=False, help="Adapter to run.")

    parser.add_argument("--config", type=Path, required=True, help="Config file.")

    return parser.parse_args()


def main():
    args = parse_args()

    solver_type = resolve("solver", args.solver)
    optimizer_type = resolve("optimizer", args.optimizer)
    adapter_type = resolve("adapter", args.adapter or None)

    with args.config.open("r") as file:
        config = json.load(file)

    config_template, tuning_parameter_space = parse_config(config)

    ifrace = IFRace(
        I=list(TRAINING_SET),
        n_x=30,
        B=2000,
    )

    ifrace.tune(
        solver_type=solver_type,
        optimizer_type=optimizer_type,
        adapter_type=adapter_type,
        config_template=config_template,
        tuning_parameter_space=tuning_parameter_space,
    )


if __name__ == "__main__":
    main()
