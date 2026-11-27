"""BSSAPSO. Everything is read from ``history["belief_space"]`` or the optimizer's
``adaptable_CPs``, except one private read (``_collapse_step``) that pins the
step at which the collapse is detected, which the history cannot show."""

import json

import numpy as np
import pytest

from edaptive.const import max_float
from edaptive.core.config import SolverConfig
from edaptive.core.parameters import Parameter, ParameterSpace, ParameterType
from edaptive.problems.ela_benchmark_suite import Rosenbrock
from edaptive.solvers.pso.bs_sapso import BSSAPSO, one_iteration_change
from edaptive.utils.experiment_utils import parse_config

NAMES = ("w", "c_1", "c_2")
LBS = np.array([0.0, 0.0, 0.0])
UBS = np.array([1.0, 4.0, 4.0])
SPACE = ParameterSpace(
    [
        Parameter(name, ParameterType.CONTINUOUS, lower=lb, upper=ub)
        for name, lb, ub in zip(NAMES, LBS, UBS, strict=True)
    ]
)
OUTSIDE = 100.0  # Rosenbrock's bounds are [-30, 30]


def settings(**overrides):
    return {
        "update_freq": 2,
        "selection": "elitist",
        "n_e": 4,
        "n_e_min": 2,
    } | overrides


def make_solver(n_s=6, space=SPACE, fixed=None, seed=0, **overrides):
    config = SolverConfig(
        identifier="bs",
        fixed_CPs={"n_s": n_s, "V_max_fraction": 1.0} | (fixed or {}),
        adapted_CP_space=space,
        adapter_params=settings(**overrides),
    )
    return BSSAPSO(problem=Rosenbrock(dimensions=2), config=config, rnd_seed=seed)


def belief_space(solver):
    """The recorded history (lists during a run)."""
    return solver.optimizer.history["belief_space"]


def CPs(solver):
    return solver.optimizer.adaptable_CPs


def set_state(solver, f_X, CPs=None, outside=()):
    """Put the swarm in a chosen state: particles in ``outside`` lie outside the
    bounds, the others at the origin."""
    X = np.zeros((solver.n_s, 2))
    X[list(outside)] = OUTSIDE
    solver.optimizer.X = X
    solver.optimizer.feasible = np.all(
        (X >= solver.optimizer.problem_lb) & (X <= solver.optimizer.problem_ub), axis=1
    )
    solver.optimizer.f_X = np.asarray(f_X, dtype=float)
    if CPs is not None:
        solver.optimizer.set_adaptable_CPs(NAMES, np.asarray(CPs, dtype=float))


def distinct_CPs(n):
    """``n`` distinct CP rows, increasing in every CP; row k is ``(k+1)/(n+1)`` of
    the way from the lower to the upper bounds."""
    fractions = np.arange(1, n + 1)[:, None] / (n + 1)
    return LBS + fractions * (UBS - LBS)


def calypso_update_iterations(p, n_iterations):
    """calypso's fixed trigger: update in iteration i when i - last > p."""
    last, fired = 0, []
    for i in range(1, n_iterations + 1):
        if i - last > p:
            last = i
            fired.append(i)
    return fired


@pytest.mark.parametrize("p", [1, 3, 7, 200])
def test_update_freq_is_the_papers_p_plus_one(p):
    n_iterations = 5000
    update_freq = p + 1
    steps = [t for t in range(n_iterations) if (t + 1) % update_freq == 0]
    # calypso's iteration i is step t = i - 1.
    assert [t + 1 for t in steps] == calypso_update_iterations(p, n_iterations)


@pytest.mark.parametrize("update_freq", [2, 4, 8])
def test_update_k_comes_after_step_k_update_freq_minus_one(update_freq):
    solver = make_solver(update_freq=update_freq, selection="improvement", n_e=6)
    resampled_after = []
    for t in range(5 * update_freq):
        before = CPs(solver)
        solver.optimizer.step(t)
        solver.adapt(t)
        if not np.array_equal(CPs(solver), before):
            resampled_after.append(t)
    assert resampled_after == [k * update_freq - 1 for k in range(1, 6)]
    assert len(belief_space(solver)["belief_lbs"]) == 5


def test_initial_CPs_are_sampled_in_the_CP_bounds():
    solver = make_solver()
    assert np.all(CPs(solver) >= LBS) and np.all(CPs(solver) <= UBS)
    assert belief_space(solver) == {
        "belief_lbs": [],
        "belief_ubs": [],
        "n_feasible": [],
    }


def test_elitist_box_is_min_max_of_the_n_e_best_particles():
    solver = make_solver(n_s=5, n_e=3)
    initial = distinct_CPs(5)
    f_X = [5.0, 1.0, 4.0, 2.0, 3.0]  # best three: particles 1, 3, 4
    set_state(solver, f_X, initial)
    solver.adapt(0)
    assert belief_space(solver)["belief_lbs"] == []
    set_state(solver, f_X, initial)
    solver.adapt(1)
    history = belief_space(solver)
    np.testing.assert_array_equal(history["belief_lbs"], [initial[1]])
    np.testing.assert_array_equal(history["belief_ubs"], [initial[4]])
    assert history["n_feasible"] == [5]
    # All particles were resampled in the new box.
    assert np.all(CPs(solver) >= initial[1]) and np.all(CPs(solver) <= initial[4])
    assert not np.array_equal(CPs(solver), initial)


def run_with_fixed_state(solver, n_steps):
    """Every step: fitness increasing with the particle index, and CPs increasing
    with it too, so the best n particles span [CPs[0], CPs[n - 1]]."""
    fixed_CPs = distinct_CPs(solver.n_s)
    for t in range(n_steps):
        set_state(solver, np.arange(solver.n_s), fixed_CPs)
        solver.adapt(t)
    return fixed_CPs


def test_elitist_selects_max_n_e_minus_k_minus_one_and_n_e_min():
    solver = make_solver(n_s=8, n_e=5, n_e_min=2)
    fixed_CPs = run_with_fixed_state(solver, 16)
    n_selected = [max(5 - (k - 1), 2) for k in range(1, 9)]  # 5, 4, 3, 2, 2, ...
    history = belief_space(solver)
    np.testing.assert_array_equal(history["belief_lbs"], fixed_CPs[[0] * 8])
    np.testing.assert_array_equal(
        history["belief_ubs"], fixed_CPs[[n - 1 for n in n_selected]]
    )


def test_improvement_selects_n_e_at_every_update():
    solver = make_solver(n_s=8, selection="improvement", n_e=5)
    # The fitness never changes, so all changes are 0 and the ties go by index.
    fixed_CPs = run_with_fixed_state(solver, 16)
    np.testing.assert_array_equal(
        belief_space(solver)["belief_ubs"], fixed_CPs[[4] * 8]
    )


def test_improvement_selects_the_largest_one_step_decreases():
    solver = make_solver(n_s=4, selection="improvement", n_e=2)
    initial = distinct_CPs(4)
    set_state(solver, [10.0, 10.0, 10.0, 10.0], initial)
    solver.adapt(0)  # f_{t-1}
    set_state(solver, [9.0, 7.0, 10.0, 8.0], initial)
    solver.adapt(1)  # changes -1, -3, 0, -2: particles 1 and 3
    np.testing.assert_array_equal(belief_space(solver)["belief_lbs"], [initial[1]])
    np.testing.assert_array_equal(belief_space(solver)["belief_ubs"], [initial[3]])


def test_one_iteration_change_infinite_cases():
    inf = np.inf
    previous = np.array([inf, 0.0, inf, 0.0, 0.0, -max_float, max_float])
    current = np.array([inf, inf, -100.0, 1.0, 2.0, max_float, -max_float])
    # inf - inf (NaN); inf - finite; finite - inf (-inf); finite changes; a
    # feasible max_float change overflowing to +inf and one to -inf.
    np.testing.assert_array_equal(
        one_iteration_change(current, previous), [inf, inf, inf, 1.0, 2.0, inf, inf]
    )


def test_improvement_ranks_infinite_changes_last():
    solver = make_solver(n_s=7, selection="improvement", n_e=2)
    initial = distinct_CPs(7)
    set_state(solver, [0, 0, 0, 0, 0, -max_float, max_float], initial, outside=(0, 2))
    solver.adapt(0)
    set_state(
        solver, [-100, -100, -100, 1, 2, max_float, -max_float], initial, outside=(0, 1)
    )
    solver.adapt(1)
    np.testing.assert_array_equal(belief_space(solver)["belief_lbs"], [initial[3]])
    np.testing.assert_array_equal(belief_space(solver)["belief_ubs"], [initial[4]])


def test_infeasible_particles_rank_last_with_ties_by_index():
    solver = make_solver(n_s=5, n_e=3)
    initial = distinct_CPs(5)
    # Particles 0, 2 and 3 have the lowest fitness but lie outside the bounds.
    f_X = [-10.0, 2.0, -5.0, -20.0, 1.0]
    set_state(solver, f_X, initial, outside=(0, 2, 3))
    solver.adapt(0)
    set_state(solver, f_X, initial, outside=(0, 2, 3))
    solver.adapt(1)
    # Both feasible particles (4, then 1), then the lowest-index infeasible one.
    history = belief_space(solver)
    np.testing.assert_array_equal(history["belief_lbs"], [initial[0]])
    np.testing.assert_array_equal(history["belief_ubs"], [initial[4]])
    assert history["n_feasible"] == [2]


def test_collapse_resamples_once_then_nothing_changes():
    solver = make_solver(n_s=4, n_e=2)
    # CPs 1e-13 apart: the box they span has widths summing below EPSILON, but
    # samples from it still differ from these CPs.
    point = np.array([0.5, 1.0, 3.0])
    near_point = point + 1e-13 * np.arange(4)[:, None]
    set_state(solver, np.arange(4), near_point)
    solver.adapt(0)
    set_state(solver, np.arange(4), near_point)
    solver.adapt(1)
    # The update collapsed the box; the CPs were still resampled once, in it.
    history = belief_space(solver)
    width = np.sum(history["belief_ubs"][0] - history["belief_lbs"][0])
    assert width < BSSAPSO.EPSILON
    after_update = CPs(solver)
    assert not np.array_equal(after_update, near_point)
    assert np.all(after_update >= history["belief_lbs"][0])
    assert np.all(after_update <= history["belief_ubs"][0])

    # From the next step on nothing changes: no update (steps 3, 5, ... are on the
    # schedule), no resampling.
    for t in range(2, 12):
        set_state(solver, np.arange(4)[::-1])
        solver.adapt(t)
    assert len(belief_space(solver)["belief_lbs"]) == 1
    np.testing.assert_array_equal(CPs(solver), after_update)
    # Private read: the collapse is detected after step 2, the step after the
    # collapsing update.
    assert solver._collapse_step == 2


def test_cps_never_change_after_a_collapse_in_a_real_run():
    update_freq, n_steps = 2, 300
    solver = make_solver(n_s=6, n_e=2, update_freq=update_freq)
    CPs_after = []  # the CPs in use from step t + 1, i.e. after adapt(t)
    for t in range(n_steps):
        solver.optimizer.step(t)
        solver.adapt(t)
        CPs_after.append(CPs(solver))

    history = belief_space(solver)
    n_updates = len(history["belief_lbs"])
    # Fewer updates than the schedule has: the run collapsed.
    assert n_updates < n_steps // update_freq
    last_box = history["belief_ubs"][-1] - history["belief_lbs"][-1]
    assert np.sum(last_box) < BSSAPSO.EPSILON
    # The last update resampled; from then on the CPs never change.
    last_update = n_updates * update_freq - 1
    assert not np.array_equal(CPs_after[last_update], CPs_after[last_update - 1])
    for CPs_t in CPs_after[last_update:]:
        np.testing.assert_array_equal(CPs_t, CPs_after[last_update])


def test_run_records_arrays():
    solver = make_solver(n_s=6, update_freq=5, selection="improvement", n_e=6)
    _, history = solver.run(max_iterations=50)
    belief_space = history["belief_space"]
    assert set(belief_space) == {"belief_lbs", "belief_ubs", "n_feasible"}
    assert (
        belief_space["belief_lbs"].shape == belief_space["belief_ubs"].shape == (10, 3)
    )
    assert belief_space["n_feasible"].shape == (10,)
    assert np.all((0 <= belief_space["n_feasible"]) & (belief_space["n_feasible"] <= 6))
    # The last box is the one the final CPs were sampled from.
    assert np.all(CPs(solver) >= belief_space["belief_lbs"][-1])
    assert np.all(CPs(solver) <= belief_space["belief_ubs"][-1])


def test_adapts_a_named_subset_in_its_own_order():
    space = ParameterSpace(
        [
            Parameter("c_2", ParameterType.CONTINUOUS, lower=0, upper=4),
            Parameter("w", ParameterType.CONTINUOUS, lower=0, upper=1),
        ]
    )
    solver = make_solver(n_s=3, space=space, fixed={"c_1": 1.5}, n_e=2)
    initial = distinct_CPs(3)
    initial[:, 1] = 1.5
    for t in (0, 1):
        set_state(solver, np.arange(3), initial)
        solver.adapt(t)
    # Particles 0 and 1 selected; the box is in (c_2, w) order.
    history = belief_space(solver)
    np.testing.assert_array_equal(history["belief_lbs"], [initial[0, [2, 0]]])
    np.testing.assert_array_equal(history["belief_ubs"], [initial[1, [2, 0]]])
    np.testing.assert_array_equal(CPs(solver)[:, 1], 1.5)


def test_timings():
    solver = make_solver()
    _, history = solver.run(max_iterations=10)
    assert history["timings"]["adapter"] == {}
    assert set(history["timings"]["solver"]) == {
        "_get_required_data",
        "_initial_masked_fitness",
        "_select",
        "_update_belief_space",
        "_sample",
        "_set_adapted_CPs",
    }


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"patience": 200}, "unexpected"),
        ({"n_elites": 4}, "unexpected"),
        ({"selection": "improve"}, "selection"),
        ({"selection": "random"}, "selection"),
        ({"update_freq": 0}, "update_freq"),
        ({"update_freq": 2.5}, "update_freq"),
        ({"n_e_min": 1, "n_e": 3}, "n_e_min"),
        ({"n_e": 1, "n_e_min": 2}, "n_e"),
    ],
)
def test_invalid_settings_raise(overrides, message):
    with pytest.raises(ValueError, match=message):
        make_solver(**overrides)


def test_missing_setting_raises():
    config = SolverConfig(
        identifier="bs",
        fixed_CPs={"n_s": 6},
        adapted_CP_space=SPACE,
        adapter_params={"update_freq": 2, "selection": "elitist", "n_e": 4},
    )
    with pytest.raises(ValueError, match="missing"):
        BSSAPSO(problem=Rosenbrock(dimensions=2), config=config)


def test_needs_continuous_adapted_CPs():
    space = ParameterSpace([Parameter("c_2", ParameterType.INTEGER, lower=0, upper=4)])
    with pytest.raises(ValueError, match="continuous"):
        make_solver(space=space)
    config = SolverConfig(
        identifier="bs", fixed_CPs={"n_s": 6}, adapter_params=settings()
    )
    with pytest.raises(ValueError, match="adapted CP space"):
        BSSAPSO(problem=Rosenbrock(dimensions=2), config=config)


def test_configs_use_update_freq_p_plus_one():
    with open("experiments/configs/bs_sapso_published.json") as file:
        published, tuning_space = parse_config(json.load(file))
    assert tuning_space is None
    # The paper's vc_fixed200_improve30: p = 200.
    assert published.adapter_params == {
        "update_freq": 201,
        "selection": "improvement",
        "n_e": 30,
        "n_e_min": 2,
    }
    with open("experiments/configs/bs_sapso.json") as file:
        _, tuning_space = parse_config(json.load(file))
    update_freq = tuning_space.get("update_freq")
    assert (update_freq.lower, update_freq.upper) == (21, 501)  # p in [20, 500]
    assert update_freq.log_scale_tuning
    assert tuning_space.get("selection").classes == ("elitist", "improvement")
