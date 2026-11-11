from edaptive.adapters.multi_valued_eda import MultiValuedEDA
from edaptive.utils.timing import timed


class RPBIL(MultiValuedEDA):
    """r-PBIL (Ben Jedidia, Doerr & Krejca 2023, Algorithm 3).

    Selects the ``mu`` best feasible particles (ties broken uniformly at random)
    and moves each frequency towards the relative frequency of that value among
    them: ``p <- (1 - rho) * p + rho * relative frequency``, with the scaling
    factor ``rho`` in [0, 1]. Then restricts the frequencies to the borders. An
    update needs at least ``mu`` feasible particles.
    """

    @timed
    def initialize(self, n_values, mu, rho, update_freq, elite_signal, rng):
        if int(mu) < 1:
            raise ValueError(f"mu must be at least 1, got {mu}")

        if not 0 <= rho <= 1:
            raise ValueError(f"rho must lie in [0, 1], got {rho}")

        self.initialize_model(n_values, update_freq, elite_signal, rng)
        self.mu = int(mu)
        self.rho = rho
        self.min_feasible = self.mu

    def select(self, scores, indices):
        return self.best(scores, indices, self.mu)

    def updated_frequencies(self, selected):
        return (1 - self.rho) * self.p + self.rho * self.relative_frequencies(selected)
