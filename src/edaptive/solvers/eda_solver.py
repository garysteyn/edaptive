from edaptive.core.solver import BaseSolver


class EDASolver(BaseSolver):
    """Adapts the CPs in ``adapted_CP_space`` with an EDA adapter.

    Once the optimizer is built, the adapter sees its initial state (``start``).
    After each step the adapter checks for an update; when it returns elites, it
    updates its model and resamples the adapted CPs of all ``n_s`` individuals.
    """

    uses_adapter = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.adapter.start(**self._get_required_data())

    def adapt(self, t):
        data = self._get_required_data()
        elite_CPs = self.adapter.check_update(**data)

        if elite_CPs is None:
            return

        self.adapter.update(elite_CPs)
        new_adapted_CPs = self.adapter.sample(N=self.n_s)
        self._set_adapted_CPs(self.adapter.CP_names, new_adapted_CPs)
