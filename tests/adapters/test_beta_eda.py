"""BetaEDA: elite selection (fitness/improvement signals, windows) and model fit."""

import numpy as np
import pytest

from edaptive.adapters.beta_eda import BetaEDA
from edaptive.core.parameters import Parameter, ParameterSpace, ParameterType

CP_NAMES = ("w", "c_1", "c_2")
N_S = 5


def make_adapter(
    elite_signal="fitness",
    n_elites=2,
    update_freq=1,
    elite_weighting=None,
    eta=0.1,
    rho_min=0.02,
    copula=None,
):
    space = ParameterSpace(
        [
            Parameter(name=name, type=ParameterType.CONTINUOUS, lower=0.0, upper=4.0)
            for name in CP_NAMES
        ]
    )
    return BetaEDA(
        CP_space=space,
        eta=eta,
        rho_min=rho_min,
        n_elites=n_elites,
        update_freq=update_freq,
        copula=copula,
        elite_weighting=elite_weighting,
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


def moment_fit(scaled, weights):
    """Method-of-moments Beta parameters from plain weighted moments."""
    mu = np.average(scaled, axis=0, weights=weights)
    var = np.sum(weights[:, None] * (scaled - mu) ** 2, axis=0)
    kappa = mu * (1 - mu) / var - 1
    return mu * kappa, (1 - mu) * kappa


# Elite CPs on [0, 4]^3, best first; scaled to [0, 1] they are ELITES / 4.
ELITES = np.array([[1.0, 2.0, 0.4], [1.4, 2.4, 1.0], [2.0, 1.6, 0.8], [1.2, 2.2, 2.0]])


@pytest.mark.parametrize(
    ("elite_weighting", "weights"),
    [
        (None, np.full(4, 1 / 4)),
        ("None", np.full(4, 1 / 4)),
        ("rank", np.array([4.0, 3.0, 2.0, 1.0]) / 10),
    ],
)
def test_update_fits_plain_weighted_moments(elite_weighting, weights):
    # eta=1 replaces the model by the new fit; a tiny rho_min keeps kappa unclipped.
    adapter = make_adapter(
        n_elites=4, elite_weighting=elite_weighting, eta=1.0, rho_min=1e-9
    )
    adapter.update(ELITES)

    a, b = moment_fit(ELITES / 4, weights)
    np.testing.assert_allclose(adapter.a, a)
    np.testing.assert_allclose(adapter.b, b)


def test_uniform_weighting_uses_ddof_0_variance():
    adapter = make_adapter(n_elites=4, eta=1.0, rho_min=1e-9)
    adapter.update(ELITES)

    scaled = ELITES / 4
    mu, var = scaled.mean(axis=0), scaled.var(axis=0, ddof=0)
    np.testing.assert_allclose(adapter.a + adapter.b, mu * (1 - mu) / var - 1)


def test_identical_elites_give_the_tightest_finite_fit():
    adapter = make_adapter(n_elites=3, eta=1.0, rho_min=0.02)
    adapter.update(np.full((3, 3), 2.0))

    assert np.all(np.isfinite(adapter.a)) and np.all(np.isfinite(adapter.b))
    np.testing.assert_allclose(adapter.a + adapter.b, 1 / 0.02 - 1)


@pytest.mark.parametrize("rho_min", [0.0, -0.1, BetaEDA.RHO_MAX, 1.0])
def test_rho_min_outside_open_unit_interval_raises(rho_min):
    with pytest.raises(ValueError, match="rho_min"):
        make_adapter(rho_min=rho_min)


def test_fewer_than_two_elites_raises():
    with pytest.raises(ValueError, match="n_elites"):
        make_adapter(n_elites=1)


def test_copula_keeps_R_when_elites_agree_on_a_CP():
    adapter = make_adapter(n_elites=4, copula="gaussian")
    a_before, b_before = adapter.a.copy(), adapter.b.copy()

    elites = ELITES.copy()
    elites[:, 0] = 0.0  # all elites at the lower bound of the first CP
    adapter.update(elites)

    np.testing.assert_array_equal(adapter.R, np.eye(3))
    assert not np.array_equal(adapter.a, a_before)
    assert not np.array_equal(adapter.b, b_before)
    assert np.all(np.isfinite(adapter.sample(10)))
