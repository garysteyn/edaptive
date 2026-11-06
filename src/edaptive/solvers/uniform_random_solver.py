from edaptive.adapters.uniform_random import UniformRandomAdapter
from edaptive.core.solver import BaseSolver


class UniformRandomSolver(BaseSolver):
    """Resamples the adapted CPs uniformly within their bounds after every step."""

    uses_adapter = True
    fixed_adapter_type = UniformRandomAdapter

    def adapt(self, t):
        new_adapted_CPs = self.adapter.sample(N=self.n_s)
        self._set_adapted_CPs(self.adapter.CP_names, new_adapted_CPs)
