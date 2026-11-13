import numpy as np

from edaptive.adapters.elite_weighting import elite_weights
from edaptive.adapters.emna import MultivariateNormalEDA
from edaptive.utils.timing import timed


class EMNAAdaptive(MultivariateNormalEDA):
    """EMNA_a (Larrañaga 2002, §4.4.2, Fig. 3.17).

    Keeps an archive (the paper's population) of the ``n_elites`` best CP vectors
    seen so far, with the scores they had when they entered it. Each feasible
    particle is a newly generated individual: it replaces the worst archive entry
    if its score is strictly lower. Applied to all feasible particles of an
    update, this keeps the ``n_elites`` best of the archive and the particles, an
    existing entry winning a tie. The first update fills the archive (the paper's
    first model, fitted to individuals selected from the first population).

    The estimate is the moments of the archive, weighted by rank on stored score
    with ``elite_weighting`` (``"rank"``, ``"log-rank"``, otherwise uniform, as in
    BetaEDA). With uniform weights these are the moments that the paper's
    incremental formulas (3.34)-(3.35) maintain. The model moves toward them with
    learning rate ``eta`` and keeps a standard deviation of at least ``sigma_min``
    in every direction (see ``MultivariateNormalEDA``).

    Stored scores are never re-evaluated, so archive entries are compared on
    scores from different iterations (with ``"improvement"``, from different
    windows). An entry with an exceptionally good early score can stay in the
    archive for the rest of the run.
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
        self.archive_scores = np.empty(0)
        self.archive_CPs = np.empty((0, self.n_CPs))

    def estimate(self, scores, unit_CPs):
        # Archive first, so that a stable sort keeps an existing entry on a tie.
        all_scores = np.concatenate([self.archive_scores, scores])
        all_CPs = np.concatenate([self.archive_CPs, unit_CPs], axis=0)
        keep = np.argsort(all_scores, kind="stable")[: self.n_elites]
        self.archive_scores = all_scores[keep]
        self.archive_CPs = all_CPs[keep]

        weights = elite_weights(len(self.archive_CPs), self.elite_weighting)
        return self.weighted_moments(self.archive_CPs, weights)
