import numpy as np
import argparse
from pathlib import Path
from edaptive.utils.experiment_utils import parse_config
from edaptive.solvers.eda_solver import EDASolver
from edaptive.solvers.basic_solver import BasicSolver
from edaptive.solvers.pso.sg_pso import stability_guided_PSO
from edaptive.adapters.beta_eda import *
from edaptive.optimizers.pso import PSO 
from edaptive.problems.ela_benchmark_suite import *
from edaptive.problems.custom_test_set import *
from mpi4py import MPI
import pickle
import json

comm = MPI.COMM_WORLD

rank = comm.Get_rank()

size = comm.Get_size()

def parse_args():
    parser = argparse.ArgumentParser(
        description=""
    )

    # We should just save the solver, optimizer, and adapter class details in the tuning directory. Then the identifier can be used to retrieve this information.
    parser.add_argument(
        "--solver",
        type=str,
        required=True,
        help="Solver to run."
    )

    parser.add_argument(
        "--optimizer",
        type=str,
        required=False,
        help="Optimizer to run."
    )

    parser.add_argument(
        "--adapter",
        type=str,
        required=False,
        help="Adapter to run."
    )

    parser.add_argument(
        "--identifier",
        type=str,
        required=True,
        help="Alias for the algorithm. Corresponds to the 'identifier' field in the tuning .json file, as well as the directory containing the tuning results."
    )

    parser.add_argument(
        "--n_x",
        type=int,
        required=False,
        default=30,
        help="Problem dimension."
    )

    parser.add_argument(
        "--max_iter",
        type=int,
        required=False,
        default=5000,
        help="Maximum no. search iterations per trial (default=30)."
    )

    parser.add_argument(
        "--n_trials",
        type=int,
        required=False,
        default=50,
        help="No. independent experiment trials per problem."
    )

    parser.add_argument(
        "--config",
        type=Path,
        required=False,
        help="Config file."
    )

    return parser.parse_args()

args = parse_args()

solver_type = eval(args.solver)

if args.optimizer:
    optimizer_type = eval(args.optimizer)
else:
    optimizer_type = None

if args.adapter:
    adapter_type = eval(args.adapter)
else:
    adapter_type = None

identifier = args.identifier

n_x = args.n_x

max_iter = args.max_iter

n_trials = args.n_trials

if args.config:
    config = args.config
    with args.config.open("r") as file:
        config = json.load(file)

tuning_results_path = Path("results/tuning_results") / identifier

config_path = tuning_results_path / "best_configuration.pkl"

was_tuned = True
if config_path.exists():
    with open(config_path, "rb") as file:
        config = pickle.load(file)
else:
    was_tuned = False
    config, _ = parse_config(config)

tuning_problems=[
            Schwefel1,
            Ripple25,
            Exponential,
            NeedleEye,
            Step3,
            GeneralizedGiunta,
            GeneralizedPaviani,
            Brown,
            CosineMixture_OG,
            CosineMixture,
            Mishra07,
            Mishra01,
            GeneralizedPrice2,
            # BBOB_FID2_IID1,
            # BBOB_FID6_IID1,
            # BBOB_FID16_IID1,
            # BBOB_FID17_IID2
        ]

evaluation_problems=[
            BBOB_FID21_IID1,
            BBOB_FID16_IID5,
            BBOB_FID23_IID5,
            BBOB_FID17_IID2,
            BBOB_FID19_IID1,
            BBOB_FID18_IID5,
            BBOB_FID19_IID4,
            BBOB_FID14_IID2,
            BBOB_FID16_IID2,
            BBOB_FID17_IID5,
            BBOB_FID1_IID4,
            BBOB_FID14_IID1,
            BBOB_FID14_IID5,
            BBOB_FID20_IID5,
            CEC_2013_FID13,
            BBOB_FID18_IID4,
            CEC_2017_FID28,
            BBOB_FID12_IID3,
            CEC_2015_FID4,
            BBOB_FID1_IID5,
            CEC_2015_FID3
        ]

# print(tuned_param_config)

local_results = {}

for i, p in enumerate(tuning_problems):

    local_results[p.__name__] = {}

    for j in range(n_trials):

        # Assign each problem-trial combination to one rank.
        task_id = i * n_trials + j

        if task_id % size != rank:
            continue

        if was_tuned:
            solver = solver_type(
                optimizer_type=optimizer_type,
                adapter_type=adapter_type,
                problem=p(dimensions=n_x),
                hyper_params=config,
                rnd_seed=j
            )
        
        else:
            if optimizer_type:      # Solver uses a general tuning-free adapter (can be applied to different kinds of optimizers)
                solver = solver_type(
                    optimizer_type=optimizer_type,
                    problem=p(dimensions=n_x),
                    hyper_params=config,
                    rnd_seed=j
                )
            else:                   # Solver uses an adaptation method tailored to a specific kind of optimizer
                solver = solver_type(
                    problem=p(dimensions=n_x),
                    hyper_params=config,
                    rnd_seed=j
                )

        _, results_dict = solver.run(max_iterations=max_iter)

        local_results[p.__name__][j] = results_dict
        
        print(
            f"Rank {rank}: {p.__name__}, trial={j}",
            flush=True,
        )

all_results = comm.gather(local_results, root=0)

if rank == 0:

    results = {
        p.__name__: {}
        for p in tuning_problems
    }

    for rank_results in all_results:
        for problem_name, trials in rank_results.items():
            results[problem_name].update(trials)

    for problem_name in results:
        results[problem_name] = dict(
            sorted(results[problem_name].items())
        )

    print([results["CosineMixture_OG"][k]["f_best"][-1] for k in range(0, 50)])

    output_path = Path("results") / "experiment_results" / f"{identifier}.pkl"

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "wb") as file:
        pickle.dump(results, file)

    print(f"Results saved to {output_path}")