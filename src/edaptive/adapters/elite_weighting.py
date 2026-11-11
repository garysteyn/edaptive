import numpy as np

# Weighting schemes other than uniform; any other value of ``elite_weighting``
# (e.g. ``None`` or ``"None"``) means uniform weights.
WEIGHTED_SCHEMES = ("rank", "log-rank")


def elite_weights(n, elite_weighting):
    """Weights of ``n`` elites (best first), summing to 1. Uniform unless
    ``elite_weighting`` is ``"rank"`` (linear, n..1) or ``"log-rank"``
    (log(n + 1) - log(rank))."""
    weights = np.full(n, 1 / n)

    if elite_weighting == "rank":
        weights = np.arange(n, 0, -1, dtype=float)

        weights /= weights.sum()

    elif elite_weighting == "log-rank":
        # ranks = 1, 2, ..., n
        ranks = np.arange(1, n + 1)

        weights = np.log(n + 1) - np.log(ranks)

        # Numerical safety (should already be positive)
        weights = np.maximum(weights, 0.0)

        weights /= weights.sum()

    return weights
