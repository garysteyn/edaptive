from edaptive.problems.ela_benchmark_suite import *
from edaptive.solvers.basic_solver import BasicSolver
from edaptive.optimizers.pso import PSO
from edaptive.core.parameters import ParameterType, Parameter, ParameterSpace
import numpy as np

def main():
    params = {
        "adapter" : None,
        "optimizer" : {
            "fixed" : {
                "n_s" : 30,
                "w" : 0.5,
                "c_1" : 1.2,
                "c_2" : 0.8
            },
        }
    }

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
        # Mishra07,
        # Mishra01,
        # GeneralizedPrice2,
        BBOB_FID2_IID1,
        BBOB_FID6_IID1,
        BBOB_FID16_IID1,
        BBOB_FID17_IID2,

    ]

    pso = BasicSolver(PSO,
                      problem=BBOB_FID17_IID2(dimensions=60),
                      hyper_params=params)
    res = pso.run(max_iterations=5000)
    
    print(res[1]["timings"], res[0])

if __name__ == "__main__":
    main()