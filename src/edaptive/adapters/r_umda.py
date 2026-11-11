from edaptive.adapters.multi_valued_eda import MultiValuedEDA
from edaptive.utils.timing import timed


class RUMDA(MultiValuedEDA):
    """r-UMDA (Ben Jedidia, Doerr & Krejca 2023, Algorithm 2).

    Selects the ``mu`` best feasible particles (ties broken uniformly at random)
    and sets each frequency to the relative frequency of that value among them,
    then restricts the frequencies to the borders. With fewer than ``mu`` feasible
    particles, all of them are selected (as in BetaEDA); an update needs at least
    one.
    """

    @timed
    def initialize(self, n_values, mu, update_freq, elite_signal, rng):
        if int(mu) < 1:
            raise ValueError(f"mu must be at least 1, got {mu}")

        self.initialize_model(n_values, update_freq, elite_signal, rng)
        self.mu = int(mu)
        self.min_feasible = 1

    def select(self, scores, indices):
        return self.best(scores, indices, self.mu)

    def updated_frequencies(self, selected):
        return self.relative_frequencies(selected)
