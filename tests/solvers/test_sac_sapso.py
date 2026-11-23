"""SACSAPSO replays a SAC policy exactly as calypso's environment presents it."""

import numpy as np
import pytest

from edaptive.core.config import SolverConfig
from edaptive.problems.ela_benchmark_suite import Rosenbrock
from edaptive.solvers.pso.sac_sapso import (
    SACSAPSO,
    build_observation,
    load_policy,
    policy_action,
    policy_distribution,
    reset_observation,
    save_policy,
)
from tests.fixtures.make_sac_sapso_policy import (
    MAX_ITERATIONS,
    N_S,
    NT,
    POLICY_PATH,
    make_policy,
)

N_X = 3


def config(policy_path=POLICY_PATH, **overrides):
    settings = {
        "policy_path": str(policy_path),
        "nt": NT,
        "max_iterations": MAX_ITERATIONS,
    }
    return SolverConfig(
        fixed_CPs={"n_s": N_S, "V_max_fraction": 1.0},
        adapter_params=settings | overrides,
    )


def build(seed=0, **kwargs):
    return SACSAPSO(
        problem=Rosenbrock(dimensions=N_X), config=config(**kwargs), rnd_seed=seed
    )


def record_decisions(solver):
    """Wrap the solver's policy call to record each observation."""
    observations = []
    original = solver._policy_action

    def recording(observation):
        observations.append(observation)
        return original(observation)

    solver._policy_action = recording
    return observations


# --- observation -------------------------------------------------------------


def test_observation_components():
    lb, ub = np.array([-1.0, 0.0]), np.array([1.0, 4.0])  # ranges 2 and 4
    V = np.array(
        [
            [0.0, 0.0],
            [1.0, -2.0],  # mean(|1/2|, |-2/4|) = 0.5
            [np.nan, 1.0],  # NaN mean -> inf -> 1
        ]
    )
    CPs = np.array([0.7298, 1.49618, 1.49618], dtype=np.float64)  # stable
    observation = build_observation(
        t=49,
        max_iterations=5000,
        infeasible_proportion=0.25,
        CPs=CPs,
        V=V,
        problem_lb=lb,
        problem_ub=ub,
    )
    assert observation.dtype == np.float64
    expected = [51 / 5000, 0.25, 1.0, 0.0, np.tanh(0.5), 1.0]
    np.testing.assert_allclose(observation, np.float64(expected))

    unstable = np.array([0.9, 4.0, 4.0], dtype=np.float64)
    observation = build_observation(49, 5000, 0.25, unstable, V, lb, ub)
    assert observation[2] == 0.0


def test_reset_observation():
    np.testing.assert_array_equal(
        reset_observation(3, 5000), np.float64([1 / 5000, 0.0, 0.5, 0, 0, 0])
    )


def test_infeasible_proportion_lags_one_step():
    solver = build()
    observations = record_decisions(solver)
    optimizer = solver.optimizer

    # Initial swarm is inside the bounds; now move 3 of 10 particles outside.
    assert solver._infeasible_proportion == 0.0
    optimizer.X[:3] = optimizer.problem_ub + 1.0
    solver.adapt(NT - 1)  # decision: sees the positions before this step's move
    assert observations[-1][1] == 0.0

    optimizer.X[:] = optimizer.problem_ub - 1e-3  # all feasible again
    solver.adapt(2 * NT - 1)
    assert observations[-1][1] == np.float64(0.3)
    solver.adapt(2 * NT)  # no decision, but records 0.0
    solver.adapt(3 * NT - 1)
    assert observations[-1][1] == 0.0


def test_decisions_every_nt_steps_and_none_after_the_last_iteration():
    solver = build()
    decided = []
    original = solver._policy_action

    step = {"t": None}
    original_adapt = solver.adapt

    def adapt(t):
        step["t"] = t
        original_adapt(t)

    def recording(observation):
        decided.append(step["t"])
        return original(observation)

    solver.adapt = adapt
    solver._policy_action = recording
    solver.run(MAX_ITERATIONS)
    assert decided == list(range(NT - 1, MAX_ITERATIONS - 1, NT))
    assert MAX_ITERATIONS - 1 not in decided


def test_actions_are_shared_and_used_from_the_next_step():
    solver = build()
    first = solver.action.copy()
    np.testing.assert_array_equal(
        solver.optimizer.adaptable_CPs, np.tile(first.astype(float), (N_S, 1))
    )
    for t in range(NT):
        solver.optimizer.step(t)
        solver.adapt(t)
        if t < NT - 1:
            np.testing.assert_array_equal(solver.action, first)
    assert not np.array_equal(solver.action, first)
    np.testing.assert_array_equal(
        solver.optimizer.adaptable_CPs, np.tile(solver.action.astype(float), (N_S, 1))
    )


def test_observation_time_and_stability_match_calypso():
    solver = build()
    observations = record_decisions(solver)
    in_force = solver.action.copy()
    for t in range(NT):
        solver.optimizer.step(t)
        solver.adapt(t)
    observation = observations[0]
    assert observation[0] == np.float64((NT - 1 + 2) / MAX_ITERATIONS)
    w, c_1, c_2 = in_force
    stable = c_1 + c_2 < 24 * (1 - w**2) / (7 - 5 * w)
    assert observation[2] == float(stable)
    V = solver.optimizer.V
    width = solver.optimizer.problem_ub - solver.optimizer.problem_lb
    np.testing.assert_allclose(
        observation[3:], np.tanh(np.mean(np.abs(V / width), axis=1)), rtol=1e-6
    )


# --- policy ------------------------------------------------------------------


def tiny_policy(tmp_path, policy_type="stochastic"):
    path = tmp_path / "tiny.npz"
    save_policy(
        path,
        hidden_layers=[([[1.0, -1.0], [0.5, 2.0]], [0.0, 0.5])],
        projection=(
            [[1.0, 0.0, 0.0, 0.5, 0.0, 0.0], [0.0, -1.0, 2.0, 0.0, 0.0, 1.0]],
            [0.0, 0.0, 0.0, 0.0, -1.0, 0.0],
        ),
        low=[0, 0, 0],
        high=[1, 4, 4],
        metadata={"policy_type": policy_type, "j": 1, "N": 0, "max_iter_cal": 1},
    )
    return load_policy(path)


def test_forward_pass_against_hand_computation(tmp_path):
    policy = tiny_policy(tmp_path)
    x = np.float64([0.2, 0.6])
    # hidden: relu([0.2*1 + 0.6*0.5 + 0, 0.2*-1 + 0.6*2 + 0.5]) = [0.5, 1.5]
    # projection: [0.5, -1.5, 3.0 | 0.25, -1.0, 1.5]
    mean, std = policy_distribution(policy, x)
    np.testing.assert_allclose(mean, [0.5, -1.5, 3.0], rtol=1e-6)
    np.testing.assert_allclose(std, np.exp([0.25, -1.0, 1.5]), rtol=1e-6)

    # Greedy: the mode, mid + half-width * tanh(mean).
    greedy = tiny_policy(tmp_path, "greedy")
    action = policy_action(greedy, x, rng=None)
    np.testing.assert_allclose(
        action,
        [0.5 + 0.5 * np.tanh(0.5), 2 + 2 * np.tanh(-1.5), 2 + 2 * np.tanh(3.0)],
        rtol=1e-6,
    )

    # Stochastic: one standard_normal(3) draw, z = mean + std * eps.
    action = policy_action(policy, x, np.random.default_rng(5))
    eps = np.random.default_rng(5).standard_normal(3).astype(np.float64)
    z = np.float64([0.5, -1.5, 3.0]) + np.exp(np.float64([0.25, -1.0, 1.5])) * eps
    expected = np.float64([0.5, 2, 2]) + np.float64([0.5, 2, 2]) * np.tanh(z)
    np.testing.assert_allclose(action, expected, rtol=1e-6)
    assert action.dtype == np.float64


def test_actions_stay_in_bounds(tmp_path):
    policy = tiny_policy(tmp_path)
    rng = np.random.default_rng(0)
    actions = np.array(
        [policy_action(policy, np.float64([5.0, 5.0]), rng) for _ in range(200)]
    )
    assert np.all(actions >= 0) and np.all(actions <= [1, 4, 4])


def test_first_action_uses_the_rng_before_the_optimizer():
    solver = build(seed=3)
    rng = np.random.default_rng(3)
    policy = load_policy(POLICY_PATH)
    expected = policy_action(policy, reset_observation(N_S, MAX_ITERATIONS), rng)
    np.testing.assert_array_equal(solver.action, expected)


# --- settings ----------------------------------------------------------------


def test_run_rejects_other_max_iterations():
    with pytest.raises(ValueError, match="max_iterations"):
        build().run(MAX_ITERATIONS + 1)


def test_settings_must_be_exactly_the_three_keys():
    with pytest.raises(ValueError, match="missing \\['nt'\\]"):
        SACSAPSO(
            problem=Rosenbrock(dimensions=N_X),
            config=SolverConfig(
                fixed_CPs={"n_s": N_S},
                adapter_params={"policy_path": str(POLICY_PATH), "max_iterations": 100},
            ),
        )
    with pytest.raises(ValueError, match="unexpected \\['eta'\\]"):
        build(eta=0.1)


@pytest.mark.parametrize(
    "overrides", [{"nt": NT + 1}, {"max_iterations": MAX_ITERATIONS + 1}]
)
def test_settings_must_match_the_policy_file(overrides):
    with pytest.raises(ValueError, match="trained with"):
        build(**overrides)


def test_swarm_size_must_match_the_policy_file():
    with pytest.raises(ValueError, match="trained with N"):
        SACSAPSO(
            problem=Rosenbrock(dimensions=N_X),
            config=SolverConfig(
                fixed_CPs={"n_s": N_S + 1},
                adapter_params={
                    "policy_path": str(POLICY_PATH),
                    "nt": NT,
                    "max_iterations": MAX_ITERATIONS,
                },
            ),
        )


# --- reproducibility ---------------------------------------------------------


def test_same_seed_reproduces_and_different_seeds_differ():
    _, a = build(seed=1).run(MAX_ITERATIONS)
    _, b = build(seed=1).run(MAX_ITERATIONS)
    _, c = build(seed=2).run(MAX_ITERATIONS)
    assert a["f_best"] == b["f_best"]
    assert a["f_best"] != c["f_best"]


def test_global_rng_untouched():
    np.random.seed(0)
    state = np.random.get_state()[1].copy()
    _, a = build(seed=1).run(MAX_ITERATIONS)
    np.testing.assert_array_equal(np.random.get_state()[1], state)
    np.random.seed(123)
    _, b = build(seed=1).run(MAX_ITERATIONS)
    assert a["f_best"] == b["f_best"]


def test_policy_interaction_is_timed():
    _, history = build().run(MAX_ITERATIONS)
    timed = history["timings"]["solver"]
    for name in (
        "_load_policy",
        "_swarm_data",
        "_observation",
        "_policy_action",
        "_set_adapted_CPs",
    ):
        assert name in timed


def test_fixture_file_matches_its_generator(tmp_path):
    make_policy(tmp_path / "policy.npz")
    regenerated = np.load(tmp_path / "policy.npz")
    checked_in = np.load(POLICY_PATH)
    assert regenerated.files == checked_in.files
    for key in checked_in.files:
        np.testing.assert_array_equal(regenerated[key], checked_in[key])
