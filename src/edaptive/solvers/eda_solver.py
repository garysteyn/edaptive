# More general
import numpy as np
from edaptive.core.solver import BaseSolver

class EDASolver(BaseSolver):
    def initialize(self, problem, hyper_params, rnd_seed=0):
        self.rng = np.random.default_rng(rnd_seed)
        self.problem = problem
        self.hyper_params = hyper_params

        adapter_params = hyper_params["adapter"]
        optimizer_params_adapted = hyper_params["optimizer"]["adapted"]
        optimizer_params_fixed = hyper_params["optimizer"]["fixed"]
        self.n_s = optimizer_params_fixed["n_s"]

        self.adapter = self.adapter_type(
            search_space=optimizer_params_adapted,
            rng=self.rng,
            **adapter_params
        )

        CP_init = self.adapter.sample(N=self.n_s)
        CP_init_dict = {param.name : CP_init[:, idx] for idx, param in enumerate(optimizer_params_adapted)}
        # print(CP_init_dict)
        self.optimizer = self.optimizer_type(
            problem=problem,
            rng=self.rng,
            **optimizer_params_fixed,
            **CP_init_dict
        )

    def run(self, max_iterations=5000):
        optimizer = self.optimizer
        adapter = self.adapter

        required_data = adapter.required_data()
        for t in range(max_iterations):
            optimizer.step(t)

            data = optimizer.get_required_data(required_data)
            samples = adapter.check_update(**data)

            if samples is None:
                continue

            adapter.update(samples)
            CP_new = adapter.sample(N=self.n_s)
            optimizer.CP = CP_new


        optimizer.history["timings"] = optimizer._timings
        return (
            np.min(optimizer.history["f_best"]),
            optimizer.history
        )