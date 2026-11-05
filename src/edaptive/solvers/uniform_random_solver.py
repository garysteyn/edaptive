import numpy as np

from edaptive.adapters.uniform_random import UniformRandomAdapter
from edaptive.core.solver import BaseSolver


class UniformRandomSolver(BaseSolver):
    def __init__(self, optimizer_type, **kwargs):
        super().__init__(optimizer_type, adapter_type=UniformRandomAdapter, **kwargs)

    def initialize(self, problem, hyper_params, rnd_seed=0):
        self.rng = np.random.default_rng(rnd_seed)
        self.problem = problem
        self.hyper_params = hyper_params

        optimizer_params_adapted = hyper_params["optimizer"]["adapted"]
        optimizer_params_fixed = hyper_params["optimizer"]["fixed"]
        self.n_s = optimizer_params_fixed["n_s"]

        self.adapter = self.adapter_type(
            CP_space=optimizer_params_adapted, rng=self.rng
        )

        initial_adapted_CPs = self.adapter.sample(N=self.n_s)
        adapted_CPs_dict = {
            name: initial_adapted_CPs[:, idx]
            for idx, name in enumerate(self.adapter.CP_names)
        }

        self.optimizer = self.optimizer_type(
            **(
                {"problem": problem}
                | {"rng": self.rng}
                | optimizer_params_fixed
                | adapted_CPs_dict
            )
        )

    def run(self, max_iterations=5000):
        optimizer = self.optimizer
        adapter = self.adapter

        for t in range(max_iterations):
            optimizer.step(t)

            new_adapted_CPs = adapter.sample(N=self.n_s)
            optimizer.set_adaptable_CPs(adapter.CP_names, new_adapted_CPs)

        optimizer.history["timings"] = optimizer._timings
        return (np.min(optimizer.history["f_best"]), optimizer.history)
