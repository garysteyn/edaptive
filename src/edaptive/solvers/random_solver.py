import numpy as np
from edaptive.core.solver import BaseSolver

from edaptive.adapters.simple_random import RandomAdapter
from edaptive.problems.ela_benchmark_suite import *
from edaptive.optimizers.pso import PSO
from edaptive.core.parameters import ParameterType, Parameter, ParameterSpace
import numpy as np

class RandomSolver(BaseSolver):
    def __init__(self, optimizer_type, **kwargs):
        super().__init__(optimizer_type, adapter_type=RandomAdapter, **kwargs)
        
    def initialize(self, problem, hyper_params, rnd_seed=0):
        self.rng = np.random.default_rng(rnd_seed)
        self.problem = problem
        self.hyper_params = hyper_params

        optimizer_params_adapted = hyper_params["optimizer"]["adapted"]
        optimizer_params_fixed = hyper_params["optimizer"]["fixed"]
        self.n_s = optimizer_params_fixed["n_s"]

        self.adapter = self.adapter_type(
            search_space=optimizer_params_adapted,
            rng=self.rng
        )

        CP_init = self.adapter.sample(N=self.n_s)
        CP_init_dict = {param.name : CP_init[:, idx] for idx, param in enumerate(optimizer_params_adapted)}

        self.optimizer = self.optimizer_type(
            **({"problem" : problem} | {"rng" : self.rng} | optimizer_params_fixed | CP_init_dict)
        )
        
    def run(self, max_iterations=5000):
        optimizer = self.optimizer
        adapter = self.adapter

        for t in range(max_iterations):
            optimizer.step(t)

            CP_new = adapter.sample(N=self.n_s)
            optimizer.CP = CP_new


        optimizer.history["timings"] = optimizer._timings
        return (
            np.min(optimizer.history["f_best"]),
            optimizer.history
        )
        
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

    params = {
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

    for problem in problems:
        solver = RandomSolver(optimizer_type=PSO, problem=problem(dimensions=10), hyper_params=params)
        res = solver.run(max_iterations=5000)
        print(res[0])

if __name__ == "__main__":
    main()