"""Elite weights shared by BetaEDA and the multi-valued EDAs."""

import numpy as np
import pytest

from edaptive.adapters.elite_weighting import elite_weights

LOG_RANK_4 = np.log([5 / 1, 5 / 2, 5 / 3, 5 / 4])


@pytest.mark.parametrize(
    ("elite_weighting", "expected"),
    [
        (None, [0.25, 0.25, 0.25, 0.25]),
        ("None", [0.25, 0.25, 0.25, 0.25]),
        ("rank", [0.4, 0.3, 0.2, 0.1]),
        ("log-rank", LOG_RANK_4 / LOG_RANK_4.sum()),
    ],
)
def test_elite_weights(elite_weighting, expected):
    weights = elite_weights(4, elite_weighting)
    np.testing.assert_allclose(weights, expected)
    assert weights.sum() == pytest.approx(1)


@pytest.mark.parametrize("elite_weighting", ["rank", "log-rank"])
def test_weights_decrease_with_rank(elite_weighting):
    assert np.all(np.diff(elite_weights(6, elite_weighting)) < 0)


@pytest.mark.parametrize("elite_weighting", [None, "rank", "log-rank"])
def test_single_elite_gets_all_weight(elite_weighting):
    np.testing.assert_allclose(elite_weights(1, elite_weighting), [1.0])
