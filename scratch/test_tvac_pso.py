import os

# One BLAS/OpenMP thread per process (set before numpy is imported).
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np

from edaptive.core.config import SolverConfig
from edaptive.problems.ela_benchmark_suite import (
    CosineMixture,
    Discus,
    GeneralizedEggCrate,
    Qing,
    Rosenbrock,
    Schwefel1,
)
from edaptive.solvers.pso.tvac_pso import TVACPSO
from edaptive.utils.timing import core_time

N_X = 30
MAX_ITERATIONS = 5000
N_RUNS = 5


def main():
    config = SolverConfig(fixed_CPs={"n_s": 30})

    problems = [
        Rosenbrock,
        Qing,
        GeneralizedEggCrate,
        Discus,
        Schwefel1,
        CosineMixture,
    ]

    for problem in problems:
        f_finals = []
        for seed in range(N_RUNS):
            pso = TVACPSO(problem=problem(dimensions=N_X), config=config, rnd_seed=seed)
            f_final, history = pso.run(max_iterations=MAX_ITERATIONS)
            f_finals.append(f_final)

            times = core_time(history["timings"])
            print(
                f"{problem.__name__:20s} seed {seed}: f = {f_final:11.4g}  "
                f"feasible = {np.mean(history['prop_feasible']):.2f}  "
                f"stable = {np.mean(history['prop_stable']):.2f}  "
                f"wall = {times['wall']:.2f} s  cpu = {times['cpu']:.2f} s"
            )

        print(
            f"{problem.__name__:20s} mean f = {np.mean(f_finals):.4g}, "
            f"median f = {np.median(f_finals):.4g}\n"
        )


if __name__ == "__main__":
    main()
