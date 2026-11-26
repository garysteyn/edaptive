import os

# One thread per process, so each process's recorded times are its own compute effort
# (set before numpy is imported; a value from the shell wins).
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
# CPU only, so CPU time covers all compute.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import argparse
import json
import pickle
from pathlib import Path

from mpi4py import MPI

from edaptive.core.config import SolverConfig
from edaptive.problems.custom_test_set import EVALUATION_SET
from edaptive.problems.ela_benchmark_suite import TRAINING_SET
from edaptive.registry import resolve
from edaptive.utils.experiment_utils import parse_config
from edaptive.utils.seeding import trial_seed

PROBLEM_SETS = {"evaluation": EVALUATION_SET, "training": TRAINING_SET}

comm = MPI.COMM_WORLD

rank = comm.Get_rank()

size = comm.Get_size()


def parse_args():
    parser = argparse.ArgumentParser(description="")

    # We should just save the solver, optimizer, and adapter class details in the
    # tuning directory. Then the identifier can be used to retrieve this information.
    parser.add_argument("--solver", type=str, required=True, help="Solver to run.")

    parser.add_argument(
        "--optimizer", type=str, required=False, help="Optimizer to run."
    )

    parser.add_argument("--adapter", type=str, required=False, help="Adapter to run.")

    parser.add_argument(
        "--identifier",
        type=str,
        required=True,
        help="Alias for the algorithm. Corresponds to the 'identifier' field in the "
        "tuning .json file, as well as the directory containing the tuning results.",
    )

    parser.add_argument(
        "--n_x", type=int, required=False, default=30, help="Problem dimension."
    )

    parser.add_argument(
        "--max_iter",
        type=int,
        required=False,
        default=5000,
        help="Maximum no. search iterations per trial (default=30).",
    )

    parser.add_argument(
        "--n_trials",
        type=int,
        required=False,
        default=50,
        help="No. independent experiment trials per problem.",
    )

    parser.add_argument("--config", type=Path, required=False, help="Config file.")

    parser.add_argument(
        "--seed_base",
        type=int,
        required=False,
        default=0,
        help="Base of the run seeds (see edaptive.utils.seeding.trial_seed).",
    )

    parser.add_argument(
        "--problem_set",
        choices=tuple(PROBLEM_SETS),
        default="evaluation",
        help="Problems to run: the ELA-selected evaluation set (custom_test_set.py) "
        "or the training set (ela_benchmark_suite.TRAINING_SET).",
    )

    return parser.parse_args()


args = parse_args()

solver_type = resolve("solver", args.solver)
optimizer_type = resolve("optimizer", args.optimizer or None)
adapter_type = resolve("adapter", args.adapter or None)

identifier = args.identifier

n_x = args.n_x

max_iter = args.max_iter

n_trials = args.n_trials

tuning_results_path = Path("results/tuning_results") / identifier

config_path = tuning_results_path / "best_configuration.pkl"

if config_path.exists():
    with open(config_path, "rb") as file:
        config = SolverConfig.from_dict(pickle.load(file))
elif args.config:
    with args.config.open("r") as file:
        config, _ = parse_config(json.load(file))
else:
    raise SystemExit(
        f"No configuration for '{identifier}': {config_path} does not exist and "
        "no --config was given."
    )

problems = PROBLEM_SETS[args.problem_set]

local_results = {}

for i, p in enumerate(problems):
    local_results[p.__name__] = {}

    for j in range(n_trials):
        # Assign each problem-trial combination to one rank.
        task_id = i * n_trials + j

        if task_id % size != rank:
            continue

        # Depends only on (problem, n_x, trial), so it is the same for every
        # algorithm and differs between problems.
        seed = trial_seed(p.__name__, n_x, j, base=args.seed_base)

        solver = solver_type(
            optimizer_type,
            adapter_type,
            problem=p(dimensions=n_x),
            config=config,
            rnd_seed=seed,
        )

        _, results_dict = solver.run(max_iterations=max_iter)
        results_dict["seed"] = seed

        local_results[p.__name__][j] = results_dict

        print(
            f"Rank {rank}: {p.__name__}, trial={j}",
            flush=True,
        )

all_results = comm.gather(local_results, root=0)

if rank == 0:
    results = {p.__name__: {} for p in problems}

    for rank_results in all_results:
        for problem_name, trials in rank_results.items():
            results[problem_name].update(trials)

    for problem_name in results:
        results[problem_name] = dict(sorted(results[problem_name].items()))

    output_path = Path("results") / "experiment_results" / f"{identifier}.pkl"

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "wb") as file:
        pickle.dump(results, file)

    print(f"Results saved to {output_path}")
