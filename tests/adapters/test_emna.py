"""EMNA adapters (Larrañaga 2002, §4.4): shared multivariate normal model and
EMNA_global."""

import numpy as np
import pytest

from edaptive.adapters.elite_weighting import elite_weights
from edaptive.adapters.emna_global import EMNAGlobal
from edaptive.core.parameters import Parameter, ParameterSpace, ParameterType

CP_NAMES = ("w", "c_1", "c_2")
LOWER = np.array([0.0, 0.0, 1.0])
UPPER = np.array([1.0, 4.0, 3.0])


def space(types=(ParameterType.CONTINUOUS,) * 3):
    return ParameterSpace(
        [
            Parameter(name=name, type=type_, lower=lower, upper=upper)
            for name, type_, lower, upper in zip(
                CP_NAMES, types, LOWER, UPPER, strict=True
            )
        ]
    )


def make_global(
    eta=1.0,
    sigma_min=1e-3,
    n_elites=3,
    update_freq=1,
    elite_weighting=None,
    elite_signal="fitness",
    seed=0,
    CP_space=None,
):
    return EMNAGlobal(
        CP_space=space() if CP_space is None else CP_space,
        eta=eta,
        sigma_min=sigma_min,
        n_elites=n_elites,
        update_freq=update_freq,
        elite_weighting=elite_weighting,
        elite_signal=elite_signal,
        rng=np.random.default_rng(seed),
    )


def data(f_X, adaptable_CPs, t=0, X=None):
    n = len(f_X)
    return {
        "problem_lb": np.zeros(2),
        "problem_ub": np.ones(2),
        "t": t,
        "X": np.full((n, 2), 0.5) if X is None else X,
        "f_X": np.asarray(f_X, dtype=float),
        "adaptable_CPs": np.asarray(adaptable_CPs, dtype=float),
        "adaptable_CP_names": CP_NAMES,
    }


def to_unit(CPs):
    return (np.asarray(CPs, dtype=float) - LOWER) / (UPPER - LOWER)


def random_CPs(n, seed=1):
    rng = np.random.default_rng(seed)
    return LOWER + rng.random((n, 3)) * (UPPER - LOWER)


# --- validation ---------------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs",
    [
        {"eta": 0.0},
        {"eta": 1.5},
        {"sigma_min": 0.0},
        {"sigma_min": 0.6},
        {"n_elites": 1},
        {"elite_signal": "rank"},
        {
            "CP_space": space(
                (
                    ParameterType.CONTINUOUS,
                    ParameterType.INTEGER,
                    ParameterType.CONTINUOUS,
                )
            )
        },
    ],
)
def test_invalid_parameters_raise(kwargs):
    with pytest.raises(ValueError):
        make_global(**kwargs)


# --- sampling -----------------------------------------------------------------


def test_initial_model_is_the_uniform_distribution():
    adapter = make_global()
    np.testing.assert_allclose(adapter.mu, 0.5)
    np.testing.assert_allclose(adapter.Sigma, np.eye(3) / 12)
    assert not adapter.fitted


def test_samples_before_the_first_update_are_uniform_in_bounds():
    adapter = make_global(seed=3)
    samples = adapter.sample(2000)
    expected = LOWER + np.random.default_rng(3).random((2000, 3)) * (UPPER - LOWER)
    np.testing.assert_allclose(samples, expected)


def test_samples_after_an_update_stay_in_bounds():
    adapter = make_global(sigma_min=0.3)
    adapter.set_model(np.zeros(3), np.eye(3) * 0.25)
    samples = adapter.sample(5000)
    assert samples.shape == (5000, 3)
    assert np.all(samples >= LOWER) and np.all(samples <= UPPER)


def test_truncated_samples_follow_the_truncated_normal():
    # Mean at the lower corner: each coordinate is a half-normal, mean sigma*sqrt(2/pi).
    adapter = make_global(sigma_min=0.01)
    adapter.set_model(np.zeros(3), np.eye(3) * 0.01)
    unit = to_unit(adapter.sample(20000))
    np.testing.assert_allclose(unit.mean(axis=0), 0.1 * np.sqrt(2 / np.pi), rtol=0.03)


def test_rejection_sampling_limit_raises():
    adapter = make_global()
    adapter.set_model(np.zeros(3), np.eye(3) * 0.01)
    adapter.mu = np.full(3, -10.0)  # far outside the box: nothing is accepted
    with pytest.raises(RuntimeError):
        adapter.sample(5)


# --- model update ---------------------------------------------------------------


def test_weighted_moments_match_hand_computation():
    rng = np.random.default_rng(2)
    X = rng.random((5, 3))
    w = elite_weights(5, "log-rank")
    mu, Sigma = EMNAGlobal.weighted_moments(X, w)
    expected_mu = sum(w[k] * X[k] for k in range(5))
    expected_Sigma = sum(
        w[k] * np.outer(X[k] - expected_mu, X[k] - expected_mu) for k in range(5)
    )
    np.testing.assert_allclose(mu, expected_mu)
    np.testing.assert_allclose(Sigma, expected_Sigma)


def test_eta_blends_toward_the_estimate():
    adapter = make_global(eta=0.3, sigma_min=1e-4)
    mu_hat = np.array([0.2, 0.4, 0.6])
    Sigma_hat = np.diag([0.01, 0.02, 0.03])
    adapter.set_model(mu_hat, Sigma_hat)
    np.testing.assert_allclose(adapter.mu, 0.7 * 0.5 + 0.3 * mu_hat)
    np.testing.assert_allclose(adapter.Sigma, 0.7 * np.eye(3) / 12 + 0.3 * Sigma_hat)
    np.testing.assert_allclose(adapter.A @ adapter.A.T, adapter.Sigma)


def test_identical_elites_give_sigma_min_in_every_direction():
    adapter = make_global(sigma_min=0.05)
    adapter.set_model(np.full(3, 0.4), np.zeros((3, 3)))
    np.testing.assert_allclose(np.linalg.eigvalsh(adapter.Sigma), 0.05**2)


def test_eigenvalues_never_fall_below_sigma_min_squared():
    adapter = make_global(sigma_min=0.1)
    # Rank-one estimate: perfectly correlated CPs.
    v = np.array([1.0, 1.0, 0.0]) / np.sqrt(2)
    adapter.set_model(np.full(3, 0.5), 0.04 * np.outer(v, v))
    eigenvalues = np.linalg.eigvalsh(adapter.Sigma)
    assert np.all(eigenvalues >= 0.1**2 - 1e-12)
    assert eigenvalues.max() == pytest.approx(0.04)


# --- check_update ---------------------------------------------------------------


def test_check_update_respects_update_freq():
    adapter = make_global(update_freq=5)
    CPs = random_CPs(4)
    assert adapter.check_update(**data([1, 2, 3, 4], CPs, t=3)) is None
    assert adapter.check_update(**data([1, 2, 3, 4], CPs, t=4)) is not None


def test_check_update_ranks_feasible_particles_in_unit_space():
    adapter = make_global()
    CPs = random_CPs(4)
    X = np.full((4, 2), 0.5)
    X[1] = 2.0  # infeasible, and it has the best fitness
    scores, unit_CPs = adapter.check_update(**data([3, 0, 1, 2], CPs, X=X))
    np.testing.assert_array_equal(scores, [1, 2, 3])
    np.testing.assert_allclose(unit_CPs, to_unit(CPs[[2, 3, 0]]))


def test_check_update_needs_two_feasible_particles():
    adapter = make_global()
    X = np.full((3, 2), 2.0)
    X[0] = 0.5
    assert adapter.check_update(**data([1, 2, 3], random_CPs(3), X=X)) is None


def test_improvement_signal_scores_the_change_and_resets_the_window():
    adapter = make_global(elite_signal="improvement")
    adapter.start(f_X=np.array([10.0, 10.0, 10.0]))
    CPs = random_CPs(3)
    scores, unit_CPs = adapter.check_update(**data([9, 5, 8], CPs))
    np.testing.assert_array_equal(scores, [-5, -2, -1])
    np.testing.assert_allclose(unit_CPs, to_unit(CPs[[1, 2, 0]]))
    scores, _ = adapter.check_update(**data([9, 5, 8], CPs))
    np.testing.assert_array_equal(scores, [0, 0, 0])


# --- EMNA_global ----------------------------------------------------------------


def test_global_with_eta_one_fits_the_elites():
    adapter = make_global(eta=1.0, n_elites=4, sigma_min=1e-3)
    CPs = random_CPs(6)
    f_X = np.array([5, 0, 3, 1, 4, 2], dtype=float)
    adapter.update(adapter.check_update(**data(f_X, CPs)))
    elites = to_unit(CPs[[1, 3, 5, 2]])
    np.testing.assert_allclose(adapter.mu, elites.mean(axis=0))
    np.testing.assert_allclose(adapter.Sigma, np.cov(elites.T, bias=True))
    assert adapter.fitted


@pytest.mark.parametrize("elite_weighting", ["rank", "log-rank"])
def test_global_weights_elites_by_rank(elite_weighting):
    adapter = make_global(n_elites=3, elite_weighting=elite_weighting)
    CPs = random_CPs(5)
    f_X = np.array([4, 2, 0, 3, 1], dtype=float)
    adapter.update(adapter.check_update(**data(f_X, CPs)))
    elites = to_unit(CPs[[2, 4, 1]])
    w = elite_weights(3, elite_weighting)
    np.testing.assert_allclose(adapter.mu, w @ elites)


def test_global_uses_all_feasible_when_fewer_than_n_elites():
    adapter = make_global(n_elites=10)
    CPs = random_CPs(3)
    adapter.update(adapter.check_update(**data([2, 1, 3], CPs)))
    np.testing.assert_allclose(adapter.mu, to_unit(CPs).mean(axis=0))


def test_global_does_not_touch_the_global_numpy_rng():
    np.random.seed(123)
    state = np.random.get_state()[1].copy()
    adapter = make_global()
    adapter.sample(10)
    adapter.update(adapter.check_update(**data([2, 1, 3, 0], random_CPs(4))))
    adapter.sample(10)
    np.testing.assert_array_equal(np.random.get_state()[1], state)
