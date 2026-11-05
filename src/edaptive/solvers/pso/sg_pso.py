import numpy as np
from edaptive.core.solver import BaseSolver
from edaptive.optimizers.pso import PSO
from scipy.integrate import cumulative_trapezoid


# Poli's stability criterion
def check_stability(w, c1, c2):
    return (c1 + c2) < (24 * (1 - w**2) / (7 - 5 * w))


class stability_guided_PSO(BaseSolver):
    adapted_CP_names = ("w", "c_1", "c_2")

    def __init__(self, **kwargs):
        super().__init__(optimizer_type=PSO, adapter_type=None, **kwargs)

    def initialize(self, problem, hyper_params, rnd_seed=0):
        self.rng = np.random.default_rng(rnd_seed)
        optimizer_params_fixed = hyper_params["optimizer"]["fixed"]
        self.n_s = optimizer_params_fixed["n_s"]

        self.adapted_CP_lbs = [0, 0, 0]
        self.adapted_CP_ubs = [1, 4, 4]

        initial_adapted_CPs = self.sample_adapted_CPs(N=self.n_s)
        adapted_CPs_dict = {
            name: initial_adapted_CPs[:, idx]
            for idx, name in enumerate(self.adapted_CP_names)
        }

        self.optimizer = self.optimizer_type(
            **(
                {"problem": problem}
                | {"rng": self.rng}
                | optimizer_params_fixed
                | adapted_CPs_dict
            )
        )

    def is_stable(self, adapted_CPs):
        column = {name: idx for idx, name in enumerate(self.adapted_CP_names)}
        return check_stability(
            adapted_CPs[:, column["w"]],
            adapted_CPs[:, column["c_1"]],
            adapted_CPs[:, column["c_2"]],
        )

    def sample_adapted_CPs(self, N):
        n_adapted_CPs = len(self.adapted_CP_names)
        adapted_CPs = self.rng.uniform(
            low=self.adapted_CP_lbs, high=self.adapted_CP_ubs, size=(N, n_adapted_CPs)
        )

        stable = self.is_stable(adapted_CPs)

        while not np.all(stable):
            adapted_CPs[~stable] = self.rng.uniform(
                low=self.adapted_CP_lbs,
                high=self.adapted_CP_ubs,
                size=(np.sum(~stable), n_adapted_CPs),
            )

            stable = self.is_stable(adapted_CPs)

        return adapted_CPs

    def run(self, max_iterations=5000):
        optimizer = self.optimizer

        for t in range(max_iterations):
            optimizer.step(t)

            new_adapted_CPs = self.sample_adapted_CPs(N=self.n_s)

            optimizer.set_adaptable_CPs(self.adapted_CP_names, new_adapted_CPs)

        optimizer.history["timings"] = optimizer._timings
        return (np.min(optimizer.history["f_best"]), optimizer.history)
