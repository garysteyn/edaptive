from edaptive.problems.ela_benchmark_suite import *
from edaptive.solvers.pso.sg_pso import stability_guided_PSO
from edaptive.core.parameters import ParameterType, Parameter, ParameterSpace
import numpy as np

def main():
    params = {
        "adapter" : None,
        "optimizer" : {
            "fixed" : {
                "n_s" : 30,
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
        CosineMixture_OG,
        # CosineMixture,
        # Mishra07,
        # Mishra01,
        # GeneralizedPrice2,
    ]

    pso = stability_guided_PSO(
                      problem=CosineMixture_OG(dimensions=30),
                      hyper_params=params)
    res = pso.run(max_iterations=5000)
    
    print(res[1]["timings"], res[0])

if __name__ == "__main__":
    main()