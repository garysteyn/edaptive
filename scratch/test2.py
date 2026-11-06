import os

# One BLAS/OpenMP thread per process (set before numpy is imported).
os.environ.setdefault("OMP_NUM_THREADS", "1")

from edaptive.problems.ela_benchmark_suite import *
from edaptive.solvers.basic_solver import BasicSolver
from edaptive.optimizers.pso import PSO
from edaptive.core.config import SolverConfig
import numpy as np

from edaptive.utils.timing import core_time


def main():
    config = SolverConfig(fixed_CPs={"n_s": 30, "w": 0.5, "c_1": 1.2, "c_2": 0.8})

    problems = [
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
        BBOB_FID2_IID1,
        BBOB_FID6_IID1,
        BBOB_FID16_IID1,
        BBOB_FID17_IID2,
    ]

    for p in problems:
        pso = BasicSolver(PSO, problem=p(dimensions=10), config=config)
        res = pso.run(max_iterations=5000)

        print(core_time(res[1]["timings"]), res[0])


if __name__ == "__main__":
    main()
