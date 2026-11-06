"""Scratch driver: runs UniformRandomSolver with PSO on a set of ELA benchmark problems."""

import os

# One BLAS/OpenMP thread per process (set before numpy is imported).
os.environ.setdefault("OMP_NUM_THREADS", "1")

from edaptive.core.config import SolverConfig
from edaptive.core.parameters import Parameter, ParameterSpace, ParameterType
from edaptive.optimizers.pso import PSO
from edaptive.problems.ela_benchmark_suite import *  # noqa: F403
from edaptive.solvers.uniform_random_solver import UniformRandomSolver


def main():
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

    config = SolverConfig(
        fixed_CPs={"n_s": 30},
        adapted_CP_space=ParameterSpace(
            [
                Parameter(name="w", lower=0.0, upper=1.0, type=ParameterType.CONTINUOUS),
                Parameter(name="c_1", lower=0.0, upper=4.0, type=ParameterType.CONTINUOUS),
                Parameter(name="c_2", lower=0.0, upper=4.0, type=ParameterType.CONTINUOUS),
            ]
        ),
    )

    for problem in problems:
        solver = UniformRandomSolver(
            PSO, problem=problem(dimensions=10), config=config
        )
        res = solver.run(max_iterations=5000)
        print(res[0])


if __name__ == "__main__":
    main()
