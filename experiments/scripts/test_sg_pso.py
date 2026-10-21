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
        # CosineMixture_OG,
        # CosineMixture,
        Mishra07,
        # Mishra01,
        # GeneralizedPrice2,
    ]

    for i in range(30):
        pso = stability_guided_PSO(
                        problem=Mishra07(dimensions=30),
                        hyper_params=params,
                        rnd_seed=i)
        res = pso.run(max_iterations=5000)
        
        print(res[1]["timings"], res[0])

if __name__ == "__main__":
    main()