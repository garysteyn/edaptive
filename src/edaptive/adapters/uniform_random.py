from edaptive.core.adapter import EDAAdapter

# TODO: Generalize the implementation to accommodate integer- and categorical-valued parameters. Currently, the implementation assumes that all parameters are continuous.


class UniformRandomAdapter(EDAAdapter):
    def initialize(self, rng):
        self.rng = rng

    def sample(self, N):
        samples = self.rng.uniform(self.CP_lbs, self.CP_ubs, size=(N, self.n_CPs))
        return samples
