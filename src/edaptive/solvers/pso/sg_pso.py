import numpy as np

from edaptive.core.solver import BaseSolver
from edaptive.optimizers.pso import PSO


# Poli's stability criterion
def check_stability(w, c1, c2):
    return (c1 + c2) < (24 * (1 - w**2) / (7 - 5 * w))


class StabilityGuidedPSO(BaseSolver):
    """Resamples w, c_1, c_2 uniformly after every step, rejecting particles that
    violate Poli's stability criterion until all of them satisfy it."""

    fixed_optimizer_type = PSO

    adapted_CP_names = ("w", "c_1", "c_2")
    adapted_CP_lbs = (0, 0, 0)
    adapted_CP_ubs = (1, 4, 4)

    def initial_adapted_CPs(self):
        initial_adapted_CPs = self.sample_adapted_CPs(N=self.n_s)
        return {
            name: initial_adapted_CPs[:, idx]
            for idx, name in enumerate(self.adapted_CP_names)
        }

    def adapt(self, t):
        new_adapted_CPs = self.sample_adapted_CPs(N=self.n_s)
        self.optimizer.set_adaptable_CPs(self.adapted_CP_names, new_adapted_CPs)

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
