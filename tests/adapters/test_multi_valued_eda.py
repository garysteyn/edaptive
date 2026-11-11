"""Multi-valued EDAs (Ben Jedidia, Doerr & Krejca 2023): shared model, r-UMDA,
r-PBIL and r-cGA."""

import numpy as np
import pytest

from edaptive.adapters.r_cga import RcGA
from edaptive.adapters.r_pbil import RPBIL
from edaptive.adapters.r_umda import RUMDA
from edaptive.core.parameters import Parameter, ParameterSpace, ParameterType

CP_NAMES = ("w", "c_1", "c_2")
N_S = 6


def space(n_CPs=3, upper=4.0):
    return ParameterSpace(
        [
            Parameter(name=name, type=ParameterType.CONTINUOUS, lower=0.0, upper=upper)
            for name in CP_NAMES[:n_CPs]
        ]
    )


def make_umda(n_values=5, mu=2, update_freq=1, elite_signal="fitness", n_CPs=3, seed=0):
    return RUMDA(
        CP_space=space(n_CPs),
        n_values=n_values,
        mu=mu,
        update_freq=update_freq,
        elite_signal=elite_signal,
        rng=np.random.default_rng(seed),
    )


def make_pbil(rho=0.5, n_values=5, mu=2, seed=0):
    return RPBIL(
        CP_space=space(),
        n_values=n_values,
        mu=mu,
        rho=rho,
        update_freq=1,
        elite_signal="fitness",
        rng=np.random.default_rng(seed),
    )


def make_cga(K=10.0, n_values=5, seed=0):
    return RcGA(
        CP_space=space(),
        n_values=n_values,
        K=K,
        update_freq=1,
        elite_signal="fitness",
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


def grid_CPs(indices):
    """CPs on the default grid (0, 1, 2, 3, 4) for value indices per particle."""
    return np.asarray(indices, dtype=float)


# --- model ------------------------------------------------------------------


def test_grid_includes_bounds():
    adapter = make_umda(n_values=5)
    assert adapter.values.shape == (3, 5)
    np.testing.assert_allclose(adapter.values[0], [0, 1, 2, 3, 4])


def test_initial_frequencies_are_uniform():
    adapter = make_umda(n_values=4)
    np.testing.assert_allclose(adapter.p, np.full((3, 4), 0.25))


def test_borders_follow_the_paper():
    adapter = make_umda(n_values=5, n_CPs=3)
    assert adapter.lower_border == pytest.approx(1 / (4 * 3))
    assert adapter.upper_border == pytest.approx(1 - 1 / 3)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"n_CPs": 1},
        {"n_CPs": 2, "n_values": 2},
        {"n_values": 1},
        {"mu": 0},
        {"elite_signal": "rank"},
    ],
)
def test_invalid_parameters_raise(kwargs):
    with pytest.raises(ValueError):
        make_umda(**kwargs)


def test_two_CPs_with_three_values_is_valid():
    adapter = make_umda(n_CPs=2, n_values=3)
    assert adapter.lower_border < 1 / 3 < adapter.upper_border


def test_restriction_sums_to_one_and_stays_in_borders():
    adapter = make_umda(n_values=5)
    rng = np.random.default_rng(1)
    p = rng.dirichlet(np.full(5, 0.2), size=3)
    p[0] = [1, 0, 0, 0, 0]
    restricted = adapter.restrict(p)
    np.testing.assert_allclose(restricted.sum(axis=1), 1)
    assert np.all(restricted >= adapter.lower_border - 1e-12)
    assert np.all(restricted <= adapter.upper_border + 1e-12)


def test_restriction_of_extreme_row_hits_both_borders():
    adapter = make_umda(n_values=5)
    restricted = adapter.restrict(np.tile([1.0, 0, 0, 0, 0], (3, 1)))
    np.testing.assert_allclose(restricted[:, 0], adapter.upper_border)
    np.testing.assert_allclose(restricted[:, 1:], adapter.lower_border)


def test_restriction_leaves_rows_inside_borders_unchanged():
    adapter = make_umda(n_values=5)
    p = np.tile([0.3, 0.2, 0.2, 0.15, 0.15], (3, 1))
    np.testing.assert_allclose(adapter.restrict(p), p)


def test_encode_inverts_sample():
    adapter = make_umda(n_values=7)
    samples = adapter.sample(50)
    indices = adapter.encode(samples)
    np.testing.assert_array_equal(adapter.values[np.arange(3), indices], samples)


def test_sample_frequencies_match_model():
    adapter = make_umda(n_values=4)
    adapter.p = np.array(
        [[0.1, 0.2, 0.3, 0.4], [0.7, 0.1, 0.1, 0.1], [0.25, 0.25, 0.25, 0.25]]
    )
    indices = adapter.encode(adapter.sample(200_000))
    np.testing.assert_allclose(
        adapter.relative_frequencies(indices), adapter.p, atol=0.01
    )


# --- integer and categorical CPs --------------------------------------------


def mixed_space(w_classes=(0.4, 0.55, 0.7298, 0.85, 1.0), c_2_upper=4):
    return ParameterSpace(
        [
            Parameter(name="w", type=ParameterType.CATEGORICAL, classes=w_classes),
            Parameter(name="c_1", type=ParameterType.CONTINUOUS, lower=0.0, upper=4.0),
            Parameter(name="c_2", type=ParameterType.INTEGER, lower=0, upper=c_2_upper),
        ]
    )


def make_mixed_umda(n_values=5, seed=0, **space_kwargs):
    return RUMDA(
        CP_space=mixed_space(**space_kwargs),
        n_values=n_values,
        mu=2,
        update_freq=1,
        elite_signal="fitness",
        rng=np.random.default_rng(seed),
    )


def test_values_follow_each_CP_type():
    adapter = make_mixed_umda()
    assert adapter.values.dtype == float
    np.testing.assert_allclose(adapter.values[0], [0.4, 0.55, 0.7298, 0.85, 1.0])
    np.testing.assert_allclose(adapter.values[1], [0, 1, 2, 3, 4])
    np.testing.assert_array_equal(adapter.values[2], [0, 1, 2, 3, 4])


def test_integer_values_span_the_range():
    adapter = make_mixed_umda(c_2_upper=6, n_values=7, w_classes=tuple(range(7)))
    np.testing.assert_array_equal(adapter.values[2], np.arange(7))


@pytest.mark.parametrize(
    "space_kwargs",
    [{"c_2_upper": 5}, {"w_classes": (0.4, 0.7, 0.9)}],
)
def test_value_count_must_equal_n_values(space_kwargs):
    with pytest.raises(ValueError, match="needs n_values = 5"):
        make_mixed_umda(**space_kwargs)


def test_mixed_space_samples_its_own_values():
    adapter = make_mixed_umda()
    samples = adapter.sample(100)
    for i in range(3):
        assert set(samples[:, i]) <= set(adapter.values[i])
    np.testing.assert_array_equal(samples[:, 2], np.round(samples[:, 2]))
    indices = adapter.encode(samples)
    np.testing.assert_array_equal(adapter.values[np.arange(3), indices], samples)


def test_string_classes_round_trip():
    adapter = make_mixed_umda(w_classes=("a", "b", "c", "d", "e"))
    assert adapter.values.dtype == object
    samples = adapter.sample(50)
    assert set(samples[:, 0]) <= {"a", "b", "c", "d", "e"}
    indices = adapter.encode(samples)
    np.testing.assert_array_equal(adapter.values[np.arange(3), indices], samples)


def test_encode_rejects_unknown_class():
    adapter = make_mixed_umda()
    with pytest.raises(ValueError, match="not among its classes"):
        adapter.encode(np.array([[0.5, 0.0, 0.0]]))


# --- r-UMDA -----------------------------------------------------------------


def test_umda_sets_relative_frequencies_of_elites_then_restricts():
    adapter = make_umda(n_values=5, mu=4)
    elites = np.array([[0, 1, 4], [0, 1, 4], [0, 2, 4], [3, 2, 4]])
    adapter.update(elites)
    expected = adapter.restrict(
        np.array(
            [
                [0.75, 0, 0, 0.25, 0],
                [0, 0.5, 0.5, 0, 0],
                [0, 0, 0, 0, 1],
            ]
        )
    )
    np.testing.assert_allclose(adapter.p, expected)
    np.testing.assert_allclose(adapter.p.sum(axis=1), 1)


def test_umda_selects_mu_lowest_fitness():
    adapter = make_umda(mu=2)
    CPs = grid_CPs([[i % 5] * 3 for i in range(N_S)])
    selected = adapter.check_update(**data([5, 0, 4, 1, 3, 2], CPs))
    assert sorted(selected[:, 0]) == [1, 3]


def test_umda_ignores_infeasible_particles():
    adapter = make_umda(mu=2)
    CPs = grid_CPs([[i % 5] * 3 for i in range(N_S)])
    X = np.full((N_S, 2), 0.5)
    X[1] = 2.0  # best fitness, but outside the bounds
    selected = adapter.check_update(**data([5, 0, 4, 1, 3, 2], CPs, X=X))
    # Particles 3 and 5 (value indices 3 and 0).
    assert sorted(selected[:, 0]) == [0, 3]


def test_umda_selects_all_feasible_particles_when_fewer_than_mu():
    adapter = make_umda(mu=3)
    X = np.full((N_S, 2), 2.0)
    X[[1, 4]] = 0.5
    CPs = grid_CPs([[i % 5] * 3 for i in range(N_S)])
    selected = adapter.check_update(**data([5, 0, 4, 1, 3, 2], CPs, X=X))
    assert sorted(selected[:, 0]) == [1, 4]


def test_umda_needs_a_feasible_particle():
    adapter = make_umda(mu=3)
    X = np.full((N_S, 2), 2.0)
    CPs = grid_CPs([[0] * 3] * N_S)
    assert adapter.check_update(**data(np.zeros(N_S), CPs, X=X)) is None


def test_umda_updates_from_one_feasible_particle():
    adapter = make_umda(mu=3)
    X = np.full((N_S, 2), 2.0)
    X[3] = 0.5
    CPs = grid_CPs([[i % 5] * 3 for i in range(N_S)])
    selected = adapter.check_update(**data(np.zeros(N_S), CPs, X=X))
    assert selected[:, 0].tolist() == [3]


def test_pbil_selects_all_feasible_particles_when_fewer_than_mu():
    adapter = make_pbil(mu=5)
    X = np.full((N_S, 2), 2.0)
    X[[0, 2, 5]] = 0.5
    CPs = grid_CPs([[i % 5] * 3 for i in range(N_S)])
    selected = adapter.check_update(**data([5, 0, 4, 1, 3, 2], CPs, X=X))
    assert sorted(selected[:, 0]) == [0, 0, 2]


def test_ties_are_broken_at_random():
    picks = set()
    for seed in range(20):
        adapter = make_umda(mu=1, seed=seed)
        CPs = grid_CPs([[i] * 3 for i in range(4)])
        selected = adapter.check_update(**data([1, 0, 0, 1], CPs))
        picks.add(int(selected[0, 0]))
    assert picks == {1, 2}


def test_update_freq_skips_iterations():
    adapter = make_umda(update_freq=3)
    CPs = grid_CPs([[0] * 3] * N_S)
    assert adapter.check_update(**data(np.zeros(N_S), CPs, t=0)) is None
    assert adapter.check_update(**data(np.zeros(N_S), CPs, t=2)) is not None


def test_improvement_signal_uses_change_since_last_update():
    adapter = make_umda(mu=1, elite_signal="improvement")
    CPs = grid_CPs([[i % 5] * 3 for i in range(N_S)])
    adapter.start(**data([10, 10, 10, 10, 1, 10], CPs))
    # Particle 4 has the lowest fitness but particle 0 improved the most.
    selected = adapter.check_update(**data([2, 9, 9, 9, 1, 9], CPs))
    assert selected[0, 0] == 0
    # The window restarted at the previous fitness.
    selected = adapter.check_update(**data([2, 9, 9, 9, 0, 9], CPs))
    assert selected[0, 0] == 4


# --- r-PBIL -----------------------------------------------------------------

ELITES = np.array([[0, 1, 4], [0, 1, 4], [0, 2, 4], [3, 2, 4]])


@pytest.mark.parametrize("rho", [-0.1, 1.1])
def test_pbil_rejects_rho_outside_unit_interval(rho):
    with pytest.raises(ValueError, match="rho"):
        make_pbil(rho=rho)


def test_pbil_is_a_convex_combination_then_restricted():
    adapter = make_pbil(rho=0.3)
    p_old = np.tile([0.3, 0.2, 0.2, 0.15, 0.15], (3, 1))
    adapter.p = p_old.copy()
    adapter.update(ELITES)
    expected = adapter.restrict(
        0.7 * p_old + 0.3 * adapter.relative_frequencies(ELITES)
    )
    np.testing.assert_allclose(adapter.p, expected)
    np.testing.assert_allclose(adapter.p.sum(axis=1), 1)


def test_pbil_with_rho_1_equals_umda():
    pbil = make_pbil(rho=1.0)
    umda = make_umda(mu=4)
    pbil.update(ELITES)
    umda.update(ELITES)
    np.testing.assert_array_equal(pbil.p, umda.p)


def test_pbil_with_rho_0_keeps_frequencies():
    adapter = make_pbil(rho=0.0)
    adapter.update(ELITES)
    np.testing.assert_allclose(adapter.p, np.full((3, 5), 0.2))


def test_pbil_selects_mu_lowest_fitness():
    adapter = make_pbil(mu=2)
    CPs = grid_CPs([[i % 5] * 3 for i in range(N_S)])
    selected = adapter.check_update(**data([5, 0, 4, 1, 3, 2], CPs))
    assert sorted(selected[:, 0]) == [1, 3]


# --- r-cGA ------------------------------------------------------------------


@pytest.mark.parametrize("K", [0.0, -1.0])
def test_cga_rejects_non_positive_K(K):
    with pytest.raises(ValueError, match="K"):
        make_cga(K=K)


def test_cga_moves_winner_up_and_loser_down_by_1_over_K():
    adapter = make_cga(K=20.0)
    adapter.p = np.tile([0.3, 0.2, 0.2, 0.15, 0.15], (3, 1))
    winner, loser = np.array([0, 1, 4]), np.array([2, 1, 3])
    expected = adapter.p.copy()
    expected[0, 0] += 1 / 20
    expected[0, 2] -= 1 / 20
    expected[2, 4] += 1 / 20
    expected[2, 3] -= 1 / 20
    adapter.update(np.array([winner, loser]))
    # All updated frequencies lie inside the borders, so the restriction keeps them.
    np.testing.assert_allclose(adapter.p, expected)
    np.testing.assert_allclose(adapter.p[1], [0.3, 0.2, 0.2, 0.15, 0.15])


def test_cga_restricts_after_update():
    adapter = make_cga(K=1.0)
    adapter.update(np.array([[0, 0, 0], [1, 1, 1]]))
    np.testing.assert_allclose(adapter.p.sum(axis=1), 1)
    assert np.all(adapter.p >= adapter.lower_border - 1e-12)
    assert np.all(adapter.p <= adapter.upper_border + 1e-12)


def test_cga_selects_two_feasible_particles_better_first():
    CPs = grid_CPs([[i % 5] * 3 for i in range(N_S)])
    f_X = [5, 0, 4, 1, 3, 2]
    X = np.full((N_S, 2), 0.5)
    X[[0, 2]] = 2.0  # infeasible
    pairs = set()
    for seed in range(50):
        adapter = make_cga(seed=seed)
        selected = adapter.check_update(**data(f_X, CPs, X=X))
        winner, loser = int(selected[0, 0]), int(selected[1, 0])
        particle = {1: 1, 3: 3, 4: 4, 0: 5}  # value index -> particle (i % 5)
        assert f_X[particle[winner]] < f_X[particle[loser]]
        pairs.add(frozenset((winner, loser)))
    # Every pair of the 4 feasible particles is drawn.
    assert len(pairs) == 6


def test_cga_needs_two_feasible_particles():
    adapter = make_cga()
    X = np.full((N_S, 2), 2.0)
    X[0] = 0.5
    CPs = grid_CPs([[0] * 3] * N_S)
    assert adapter.check_update(**data(np.zeros(N_S), CPs, X=X)) is None


def test_cga_breaks_ties_at_random():
    winners = set()
    for seed in range(20):
        adapter = make_cga(seed=seed)
        CPs = grid_CPs([[0] * 3, [1] * 3])
        selected = adapter.check_update(**data([1.0, 1.0], CPs))
        winners.add(int(selected[0, 0]))
    assert winners == {0, 1}
