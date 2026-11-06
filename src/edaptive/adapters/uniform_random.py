from edaptive.core.adapter import EDAAdapter
from edaptive.utils.timing import timed

# TODO: Generalize the implementation to accommodate integer- and categorical-valued
# parameters. Currently, the implementation assumes that all parameters are continuous.


class UniformRandomAdapter(EDAAdapter):
    @timed
    def initialize(self, rng):
        self.rng = rng

    @timed
    def sample(self, N):
        samples = self.rng.uniform(self.CP_lbs, self.CP_ubs, size=(N, self.n_CPs))
        return samples
