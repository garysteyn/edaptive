"""BetaEDA elite selection: fitness vs improvement signals and update windows."""

import numpy as np
import pytest

from edaptive.adapters.beta_eda import BetaEDA
from edaptive.core.parameters import Parameter, ParameterSpace, ParameterType

CP_NAMES = ("w", "c_1", "c_2")
N_S = 5


def make_adapter(elite_signal="fitness", n_elites=2, update_freq=1):
    space = ParameterSpace(
        [
            Parameter(name=name, type=ParameterType.CONTINUOUS, lower=0.0, upper=4.0)
            for name in CP_NAMES
        ]
    )
    return BetaEDA(
        CP_space=space,
        eta=0.1,
        rho_min=0.02,
        n_elites=n_elites,
        update_freq=update_freq,
        copula=None,
        elite_weighting=None,
        elite_signal=elite_signal,
        rng=np.random.default_rng(0),
    )


def data(f_X, t=0, X=None):
    """Optimizer data for N_S particles whose CPs are (i, i, i) for particle i."""
    return {
        "problem_lb": np.zeros(2),
        "problem_ub": np.ones(2),
        "t": t,
        "X": np.full((N_S, 2), 0.5) if X is None else X,
        "f_X": np.asarray(f_X, dtype=float),
        "adaptable_CPs": np.repeat(np.arange(N_S, dtype=float)[:, None], 3, axis=1),
        "adaptable_CP_names": CP_NAMES,
    }


def particles(elite_CPs):
    return sorted(int(i) for i in elite_CPs[:, 0])


def test_fitness_signal_picks_lowest_fitness():
    adapter = make_adapter("fitness")
    adapter.start(**data([0, 0, 0, 0, 0]))
    elites = adapter.check_update(**data([5, 1, 4, 2, 3]))
    assert particles(elites) == [1, 3]


def test_improvement_signal_picks_largest_decrease():
    adapter = make_adapter("improvement")
    adapter.start(**data([10, 2, 9, 3, 100]))
    # Decreases: 9, 1, 5, 1, 97.
    elites = adapter.check_update(**data([1, 1, 4, 2, 3]))
    assert particles(elites) == [0, 4]


def test_improvement_window_restarts_only_when_elites_are_returned():
    adapter = make_adapter("improvement", update_freq=2)
    adapter.start(**data([10, 10, 10, 10, 10]))

    # Off the update_freq boundary: no update, window unchanged.
    assert adapter.check_update(**data([0, 0, 0, 0, 0], t=0)) is None
    np.testing.assert_array_equal(adapter.f_X_window_start, [10] * N_S)

    # Fewer than 2 feasible particles: no update, window unchanged.
    X = np.full((N_S, 2), 2.0)
    X[0] = 0.5
    assert adapter.check_update(**data([0, 0, 0, 0, 0], t=1, X=X)) is None
    np.testing.assert_array_equal(adapter.f_X_window_start, [10] * N_S)

    # Update: elites come from the window since start, then the window restarts.
    elites = adapter.check_update(**data([9, 4, 8, 6, 5], t=3))
    assert particles(elites) == [1, 4]
    np.testing.assert_array_equal(adapter.f_X_window_start, [9, 4, 8, 6, 5])


def test_infeasible_particles_are_never_elites():
    adapter = make_adapter("fitness")
    X = np.full((N_S, 2), 0.5)
    X[1] = -1.0
    elites = adapter.check_update(**data([5, 0, 4, 2, 3], X=X))
    assert particles(elites) == [3, 4]


def test_invalid_elite_signal_raises():
    with pytest.raises(ValueError, match="elite_signal"):
        make_adapter("pbest")
