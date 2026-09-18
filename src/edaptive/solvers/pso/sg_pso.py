import numpy as np
from edaptive.core.solver import BaseSolver
from edaptive.optimizers.pso import PSO
from scipy.integrate import cumulative_trapezoid

# Joint uniform sampling of PSO control parameters over the stability-feasible region using inverse transform sampling
class JointStabilitySampler:
    def __init__(self, num_grid_points=1000):
        self.w_grid = np.linspace(0.0, 1.0, num_grid_points)

        # Cross-sectional max sum
        c_sum_max = (
            24.0 * (1.0 - self.w_grid**2)
            / (7.0 - 5.0 * self.w_grid)
        )

        # Cross-sectional area of the feasible (c1, c2) triangle
        area = 0.5 * c_sum_max**2

        # Cumulative volume CDF
        cdf = cumulative_trapezoid(
            area,
            self.w_grid,
            initial=0.0,
        )
        self.cdf_w = cdf / cdf[-1]

    def _calc_c_sum_max(self, w):
        return (
            24.0 * (1.0 - w**2)
            / (7.0 - 5.0 * w)
        )

    def sample(self, N, rng=None):

        U = rng.uniform(0.0, 1.0, size=(N, 3))

        # Sample w according to volume density
        w = np.interp(
            U[:, 0],
            self.cdf_w,
            self.w_grid,
        )

        # Compute max sum at sampled w values
        c_sum_max = self._calc_c_sum_max(w)

        # Sample uniformly over the feasible triangle
        c_sum = c_sum_max * np.sqrt(U[:, 1])

        c_1 = c_sum * (1.0 - U[:, 2])
        c_2 = c_sum * U[:, 2]

        return np.column_stack((w, c_1, c_2))

class stability_guided_PSO(BaseSolver):
    def __init__(self, **kwargs):
        super().__init__(optimizer_type=PSO, adapter_type=None, **kwargs)

    def initialize(self, problem, hyper_params, rnd_seed=0):
        self.rng = np.random.default_rng(rnd_seed)
        optimizer_params_fixed = hyper_params["optimizer"]["fixed"]
        self.n_s = optimizer_params_fixed["n_s"]

        self.sampler = JointStabilitySampler()
        
        CP_init = self.sampler.sample(N=self.n_s, rng=self.rng)
        print(CP_init)
        CP_init_dict = {"w" : CP_init[:, 0], "c_1" : CP_init [:, 1], "c_2": CP_init[:, 2]}

        self.optimizer = self.optimizer_type(
            **({"problem" : problem} | {"rng" : self.rng} | optimizer_params_fixed | CP_init_dict)
        )

    def run(self, max_iterations=5000):
        optimizer = self.optimizer
        
        for t in range(max_iterations):
            optimizer.step(t)

            CP_new = self.sampler.sample(N=self.n_s, rng=self.rng)
            optimizer.CP = CP_new

        optimizer.history["timings"] = optimizer._timings
        return (
            np.min(optimizer.history["f_best"]),
            optimizer.history
        )
