import os

# One BLAS/OpenMP thread per process (set before numpy is imported).
os.environ.setdefault("OMP_NUM_THREADS", "1")

from edaptive.problems.ela_benchmark_suite import *
from edaptive.solvers.eda_solver import EDASolver
from edaptive.optimizers.pso import PSO

# from edaptive.adapters.beta_eda import BetaMarginalsEDA, BetaMarginals_log_rank_weights_EDA, BetaMarginals_rank_weights_EDA
from edaptive.adapters.beta_eda import BetaEDA
from edaptive.core.config import SolverConfig
from edaptive.core.parameters import ParameterType, Parameter, ParameterSpace
import numpy as np

from edaptive.utils.timing import core_time

problems = [
    # Schwefel1,
    # Ripple25,
    # Exponential,
    # NeedleEye,
    # Step3,
    # GeneralizedGiunta,
    # GeneralizedPaviani,
    # Brown,
    # CosineMixture_OG,
    # CosineMixture,
    Mishra07,
    Mishra07,
    Mishra07,
    # Mishra01,
    # GeneralizedPrice2,
    # BBOB_FID2_IID1,
    # BBOB_FID6_IID1,
    # BBOB_FID16_IID1,
    # BBOB_FID17_IID2
]


def main():
    config = SolverConfig(
        adapter_params={
            "eta": 0.001,
            "rho_min": 0.02,
            "n_elites": 30,
            "update_freq": 1,
            "copula": None,
            "elite_weighting": "rank",
        },
        fixed_CPs={"n_s": 30},
        adapted_CP_space=ParameterSpace(
            [
                Parameter(name="w", lower=0.0, upper=1.0, type=ParameterType.CONTINUOUS),
                Parameter(name="c_1", lower=0.0, upper=4.0, type=ParameterType.CONTINUOUS),
                Parameter(name="c_2", lower=0.0, upper=4.0, type=ParameterType.CONTINUOUS),
            ]
        ),
    )

    for i, problem in enumerate(problems):
        # print(problem)
        eda_pso = EDASolver(
            PSO,
            BetaEDA,
            problem=problem(dimensions=30),
            config=config,
            rnd_seed=i,
        )

        res = eda_pso.run()
        print(core_time(res[1]["timings"]), res[0])
        # print(res[1]["prop_stable"])
        # print(eda_pso.metaheuristic.f_Y_hat, eda_pso.metaheuristic.X.mean())
        # print(eda_pso.metaheuristic.history["timings"])
        # print(type(eda_pso.metaheuristic.history))


if __name__ == "__main__":
    main()
