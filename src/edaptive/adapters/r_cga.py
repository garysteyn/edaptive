import numpy as np

from edaptive.adapters.multi_valued_eda import MultiValuedEDA
from edaptive.utils.timing import timed


class RcGA(MultiValuedEDA):
    """r-cGA (Ben Jedidia, Doerr & Krejca 2023, Algorithm 4).

    Draws two distinct feasible particles uniformly at random as the two
    samples, ranks them by score (a tie is broken uniformly at random) and moves
    each CP's frequencies by ``1/K``: up for the winner's value, down for the
    loser's (no change where both have the same value). Then restricts the
    frequencies to the borders. ``K`` > 0 is the hypothetical population size.
    """

    @timed
    def initialize(self, n_values, K, update_freq, elite_signal, rng):
        if not K > 0:
            raise ValueError(f"K must be positive, got {K}")

        self.initialize_model(n_values, update_freq, elite_signal, rng)
        self.K = K
        self.min_feasible = 2

    def select(self, scores, indices):
        """Value indices of the winner and the loser, shape ``(2, n_CPs)``."""
        first, second = self.rng.choice(len(scores), size=2, replace=False)
        if scores[second] < scores[first] or (
            scores[second] == scores[first] and self.rng.random() < 0.5
        ):
            first, second = second, first
        return indices[[first, second]]

    def updated_frequencies(self, selected):
        winner, loser = selected
        values = np.arange(self.n_values)
        step = (winner[:, None] == values).astype(float) - (loser[:, None] == values)
        return self.p + step / self.K
