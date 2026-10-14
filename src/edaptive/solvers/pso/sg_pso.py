import numpy as np
from edaptive.core.solver import BaseSolver
from edaptive.optimizers.pso import PSO
from scipy.integrate import cumulative_trapezoid

# Poli's stability criterion
def check_stability(w, c1, c2):
    return (c1 + c2) < (24 * (1 - w**2) / (7 - 5 * w))

class stability_guided_PSO(BaseSolver):
    def __init__(self, **kwargs):
        super().__init__(optimizer_type=PSO, adapter_type=None, **kwargs)

    def initialize(self, problem, hyper_params, rnd_seed=0):
        self.rng = np.random.default_rng(rnd_seed)
        optimizer_params_fixed = hyper_params["optimizer"]["fixed"]
        self.n_s = optimizer_params_fixed["n_s"]

        self.CP_lb = [0, 0, 0]
        self.CP_ub = [1, 4, 4]
        
        CP_init = self.sample_CP(N=self.n_s)
        CP_init_dict = {"w" : CP_init[:, 0], "c_1" : CP_init [:, 1], "c_2": CP_init[:, 2]}

        self.optimizer = self.optimizer_type(
            **({"problem" : problem} | {"rng" : self.rng} | optimizer_params_fixed | CP_init_dict)
        )

    def sample_CP(self, N):
        CP = self.rng.uniform(low=self.CP_lb, high=self.CP_ub, size=(N, 3))

        stable = check_stability(CP[:, 0], CP[:, 1], CP[:, 2])

        while not np.all(stable):
            CP[~stable] = self.rng.uniform(
                low=self.CP_lb,
                high=self.CP_ub,
                size=(np.sum(~stable), 3),
            )

            stable = check_stability(CP[:, 0], CP[:, 1], CP[:, 2])

        return CP

    def run(self, max_iterations=5000):
        optimizer = self.optimizer
        
        for t in range(max_iterations):
            optimizer.step(t)

            CP_new = self.sample_CP(N=self.n_s)

            optimizer.CP = CP_new

        optimizer.history["timings"] = optimizer._timings
        return (
            np.min(optimizer.history["f_best"]),
            optimizer.history
        )
