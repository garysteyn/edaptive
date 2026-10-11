import numpy as np
from edaptive.core.adapter import BaseAdapter
from scipy.stats import beta, norm

# TODO: Generalize the implementation to accommodate integer- and categorical-valued parameters. Currently, the implementation assumes that all parameters are continuous.

class RandomAdapter(BaseAdapter):
    def initialize(self, rng):
        self.rng = rng

    def sample(self, N):
        samples = self.rng.uniform(self.CP_lb, self.CP_ub, size=(N, self.n_x))
        return samples