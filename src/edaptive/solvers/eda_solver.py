from edaptive.core.solver import BaseSolver


class EDASolver(BaseSolver):
    """Adapts the CPs in ``adapted_CP_space`` with an EDA adapter.

    After each step the adapter checks for an update; when it returns elites, it
    updates its model and resamples the adapted CPs of all ``n_s`` individuals.
    """

    uses_adapter = True

    def adapt(self, t):
        data = self.optimizer.get_required_data(self.adapter.required_data())
        elite_CPs = self.adapter.check_update(**data)

        if elite_CPs is None:
            return

        self.adapter.update(elite_CPs)
        new_adapted_CPs = self.adapter.sample(N=self.n_s)
        self.optimizer.set_adaptable_CPs(self.adapter.CP_names, new_adapted_CPs)
