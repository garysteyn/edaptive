from edaptive.problems.ela_benchmark_suite import *
from edaptive.solvers.pso.sg_pso import StabilityGuidedPSO
from edaptive.core.config import SolverConfig
import numpy as np


def main():
    config = SolverConfig(fixed_CPs={"n_s": 30})

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
        # Mishra01,
        # GeneralizedPrice2,
    ]

    for i in range(30):
        pso = StabilityGuidedPSO(
            problem=Mishra07(dimensions=30), config=config, rnd_seed=i
        )
        res = pso.run(max_iterations=5000)

        print(res[1]["timings"], res[0])


if __name__ == "__main__":
    main()
