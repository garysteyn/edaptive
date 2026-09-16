import numpy as np
from edaptive.core.solver import BaseSolver

class BasicSolver(BaseSolver):

    def initialize(self, problem, hyper_params, rnd_seed=0):
        self.rng = np.random.default_rng(rnd_seed)

        self.optimizer = self.optimizer_type(
            **({"problem" : problem} | {"rng" : self.rng} | hyper_params["optimizer"]["fixed"])
        )

    def run(self, max_iterations=5000):
        optimizer = self.optimizer
        
        for t in range(max_iterations):
            optimizer.step(t)

        optimizer.history["timings"] = optimizer._timings
        return (
            np.min(optimizer.history["f_best"]),
            optimizer.history
        )
