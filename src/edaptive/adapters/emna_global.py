from edaptive.adapters.elite_weighting import elite_weights
from edaptive.adapters.emna import MultivariateNormalEDA
from edaptive.utils.timing import timed


class EMNAGlobal(MultivariateNormalEDA):
    """EMNA_global (Larrañaga 2002, §4.4.1, Fig. 3.16).

    Each update selects the ``n_elites`` best feasible particles and estimates the
    mean and covariance of their CPs (eqs. (3.31)-(3.33)), weighted by rank with
    ``elite_weighting`` (``"rank"``, ``"log-rank"``, otherwise uniform, as in
    BetaEDA; uniform weights are the paper's estimate). With fewer than
    ``n_elites`` feasible particles, all of them are selected (as in BetaEDA); an
    update needs at least two. The model moves toward the estimate with learning
    rate ``eta`` and keeps a standard deviation of at least ``sigma_min`` in every
    direction (see ``MultivariateNormalEDA``).
    """

    @timed
    def initialize(
        self,
        eta,
        sigma_min,
        n_elites,
        update_freq,
        elite_weighting,
        elite_signal,
        rng,
    ):
        self.initialize_model(
            eta, sigma_min, n_elites, update_freq, elite_weighting, elite_signal, rng
        )

    def estimate(self, scores, unit_CPs):
        elites = unit_CPs[: self.n_elites]
        weights = elite_weights(len(elites), self.elite_weighting)
        return self.weighted_moments(elites, weights)
