import numpy as np
from edaptive.problems.ela_benchmark_suite import *
from edaptive.solvers.eda_solver import EDASolver
from edaptive.optimizers.pso import PSO
from edaptive.adapters.beta_eda import BetaMarginalsEDA

from edaptive.core.parameters import ParameterType, Parameter, ParameterSpace

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
    BBOB_FID17_IID2
    
]

def main():
    params = {
        "adapter" : {
            "eta" : 0.001,
            "kappa_max" : 50,
            "n_elite" : 15
        },

        "optimizer" : {
            "fixed" : {
                "n_s" : 30,
            },
            "adapted" : ParameterSpace([
                Parameter(
                    name="w",
                    lower=0.0,
                    upper=1.0,
                    type=ParameterType.CONTINUOUS
                ),
                Parameter(
                    name="c_1",
                    lower=0.0,
                    upper=4.0,
                    type=ParameterType.CONTINUOUS
                ),
                Parameter(
                    name="c_2",
                    lower=0.0,
                    upper=4.0,
                    type=ParameterType.CONTINUOUS
                )
            ])
        }
    }


    for i, problem in enumerate(problems):
        # print(problem)
        eda_pso = EDASolver(optimizer_type=PSO, adapter_type=BetaMarginalsEDA)
        # eda_pso = EDASolver(optimizer_type=PSO, adapter_type=BetaMarginalsEDA)
        eda_pso.initialize(
            problem=problem(dimensions=40),
            hyper_params = params
        )

        res = eda_pso.run()
        print(res[1]["timings"], res[0])
        # print(res[1]["prop_stable"])
        # print(eda_pso.metaheuristic.f_Y_hat, eda_pso.metaheuristic.X.mean())
        # print(eda_pso.metaheuristic.history["timings"])
        # print(type(eda_pso.metaheuristic.history))

if __name__ == "__main__":
    main()