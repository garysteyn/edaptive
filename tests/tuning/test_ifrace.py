"""Unit tests for the pieces of the elitist I/F-race tuner (``tuning/ifrace.py``)."""

import itertools
import json
import time

import numpy as np
import pytest

from edaptive.core.parameters import Parameter, ParameterSpace, ParameterType
from edaptive.tuning.ifrace import (
    CONFIG_ID,
    IFRace,
    IFRaceSamplingModel,
    _ceil_to_multiple,
    _round_real,
    _TuningLogger,
)

CONTINUOUS = Parameter("x", ParameterType.CONTINUOUS, lower=0.0, upper=1.0)
LOG_CONTINUOUS = Parameter(
    "eta", ParameterType.CONTINUOUS, lower=1e-4, upper=0.2, log_scale_tuning=True
)
INTEGER = Parameter("n", ParameterType.INTEGER, lower=2, upper=5)
LOG_INTEGER = Parameter(
    "m", ParameterType.INTEGER, lower=1, upper=1000, log_scale_tuning=True
)
CATEGORICAL = Parameter("kind", ParameterType.CATEGORICAL, classes=("a", "b", "c"))
SPACE = ParameterSpace([CONTINUOUS, LOG_CONTINUOUS, INTEGER, LOG_INTEGER, CATEGORICAL])


class ProblemA:
    pass


class ProblemB:
    pass


class ProblemC:
    pass


def assert_in_space(configuration):
    for param in SPACE:
        value = configuration[param.name]
        if param.type is ParameterType.CATEGORICAL:
            assert value in param.classes
        else:
            assert param.lower <= value <= param.upper, param.name
            if param.type is ParameterType.INTEGER:
                assert isinstance(value, int)


def make_tuner(**kwargs):
    defaults = {"I": [ProblemA, ProblemB, ProblemC], "n_x": 2, "B": 100}
    return IFRace(**(defaults | kwargs))


def with_elites(tuner, elites, histories, n_min=2):
    """Put a tuner in the state it has after a race (elites and their histories)."""
    tuner._elites = elites
    tuner._elite_histories = histories
    tuner._n_min = n_min
    tuner._rng = np.random.default_rng(0)
    return tuner


# --- formulas -----------------------------------------------------------------


@pytest.mark.parametrize(
    "x, d, expected", [(1, 1, 1), (5, 1, 5), (5, 2, 6), (6, 2, 6), (7, 3, 9)]
)
def test_ceil_to_multiple(x, d, expected):
    assert _ceil_to_multiple(x, d) == expected


def test_n_new_first_race():
    # floor(B_1 / (mu + T_each * min(5, 1))) = floor(100 / (5 + 1)) = 16
    assert make_tuner()._n_new_configurations(1, 100) == 16


def test_n_new_elitist_race_uses_eq_5():
    tuner = with_elites(
        make_tuner(),
        elites=[{CONFIG_ID: "a"}, {CONFIG_ID: "b"}],
        histories=[{(ProblemA, 1): 0.0, (ProblemB, 2): 0.0}, {(ProblemA, 1): 0.0}],
    )
    # e = 2, N_elite = 2: N_l = floor((50 + 2 * 2) / max(5 + 2, nm(1 + 2, 1))) = 7
    assert tuner._n_new_configurations(2, 50) == 7 - 2


def test_n_new_long_elite_history_dominates():
    tuner = with_elites(
        make_tuner(T_each=2),
        elites=[{CONFIG_ID: "a"}],
        histories=[{(ProblemA, s): 0.0 for s in range(10)}],
    )
    # e = 10: N_l = floor((40 + 10) / max(5 + 2 * 2, nm(11, 2) = 12)) = 4
    assert tuner._n_new_configurations(2, 40) == 4 - 1


def test_n_new_never_negative_when_budget_is_tiny():
    tuner = with_elites(
        make_tuner(),
        elites=[{CONFIG_ID: "a"}, {CONFIG_ID: "b"}],
        histories=[{(ProblemA, 1): 0.0}, {(ProblemA, 1): 0.0}],
    )
    assert tuner._n_new_configurations(2, 1) == 0


# --- elite selection ------------------------------------------------------------


def test_select_elites_ranks_weights_and_merges_history():
    old_elite = {"x": 0.5, CONFIG_ID: "config_1"}
    new_a = {"x": 0.1, CONFIG_ID: "config_7"}
    new_b = {"x": 0.9, CONFIG_ID: "config_8"}
    tuner = with_elites(
        make_tuner(),
        elites=[old_elite],
        histories=[{(ProblemC, 3): 9.0, (ProblemA, 1): 9.0}],
        n_min=2,
    )
    instances = [(ProblemA, 1), (ProblemB, 2)]
    # Minimisation: new_b is best on both instances, then old_elite, then new_a.
    results = [[2.0, 2.0], [3.0, 3.0], [1.0, 1.0]]
    tuner._select_elites([old_elite, new_a, new_b], results, instances)

    assert tuner.elites == [new_b, old_elite]
    # eq. 1 with N_elite = 2: (2 - r + 1) / 3
    np.testing.assert_allclose(tuner.elite_weights, [2 / 3, 1 / 3])
    assert tuner._elite_histories[0] == {(ProblemA, 1): 1.0, (ProblemB, 2): 1.0}
    # The old elite keeps earlier results; this race's result overwrites (ProblemA, 1).
    assert tuner._elite_histories[1] == {
        (ProblemC, 3): 9.0,
        (ProblemA, 1): 2.0,
        (ProblemB, 2): 2.0,
    }


def test_select_elites_keeps_at_most_n_min():
    configs = [{CONFIG_ID: f"c{i}"} for i in range(5)]
    tuner = with_elites(make_tuner(), None, None, n_min=3)
    tuner._elites = None
    tuner._select_elites(configs, [[float(i)] for i in range(5)], [(ProblemA, 0)])
    assert [c[CONFIG_ID] for c in tuner.elites] == ["c0", "c1", "c2"]
    np.testing.assert_allclose(tuner.elite_weights.sum(), 1.0)


# --- instance order -------------------------------------------------------------


def take(problems, n):
    return list(itertools.islice(problems, n))


def test_first_race_cycles_through_shuffled_instances_with_fresh_seeds():
    tuner = with_elites(make_tuner(), None, None)
    problems, e = tuner._instance_sequence(1)
    assert e == 0
    first, second = take(problems, 3), take(problems, 3)
    for block in (first, second):
        assert sorted(p.__name__ for p, _ in block) == [
            "ProblemA",
            "ProblemB",
            "ProblemC",
        ]
    # A repeated instance gets a new seed.
    assert len({seed for _, seed in first + second}) == 6


def test_later_race_order_new_then_seen_then_more_new():
    tuner = with_elites(
        make_tuner(I=[ProblemA, ProblemB, ProblemC], T_new=1), None, None
    )
    stream = take(tuner._unseen_problems(0), 6)
    # Race 1 raced the first two stream entries; they are the elite's history.
    for instance, seed in stream[:2]:
        tuner._mark_seen(instance, seed)
    seen = {problem: 0.0 for problem in stream[:2]}
    with_elites(tuner, elites=[{CONFIG_ID: "x"}], histories=[seen])

    problems, e = tuner._instance_sequence(2)
    assert e == 2
    order = take(problems, 7)
    # One new stream entry first, then the elite's entries, then further new ones.
    assert order[0] == stream[2]
    assert set(order[1:3]) == set(stream[:2])
    assert order[3:6] == stream[3:6]
    # The stream never runs out: it continues with a fresh block.
    assert len(tuner._problem_stream) == 9
    assert order[6] == tuner._problem_stream[6]


def test_later_race_replays_every_elite_instance():
    tuner = with_elites(make_tuner(T_new=1), None, None)
    stream = take(tuner._unseen_problems(0), 3)
    for problem in stream:
        tuner._mark_seen(*problem)
    # The second elite is the only one that saw stream[2].
    histories = [
        {stream[0]: 0.0, stream[1]: 0.0},
        {stream[0]: 0.0, stream[2]: 0.0},
    ]
    with_elites(tuner, [{CONFIG_ID: "x"}, {CONFIG_ID: "y"}], histories)

    problems, e = tuner._instance_sequence(2)
    assert e == 2
    order = take(problems, 4)
    assert order[0] not in stream
    assert set(order[1:4]) == set(stream)


def test_only_the_next_unseen_entry_advances_the_stream():
    tuner = with_elites(make_tuner(T_new=2), None, None)
    stream = take(tuner._unseen_problems(0), 3)
    tuner._mark_seen(*stream[0])
    # Only an entry at the first unseen index advances it.
    tuner._mark_seen(*stream[2])
    assert tuner._n_problems_seen == 1
    tuner._mark_seen(*stream[1])
    assert tuner._n_problems_seen == 2


# --- caching and evaluation -------------------------------------------------------


def test_elites_reuse_cached_results_after_new_instances():
    elite = {CONFIG_ID: "e"}
    new = {CONFIG_ID: "n"}
    tuner = with_elites(
        make_tuner(T_new=1), elites=[elite], histories=[{(ProblemA, 5): 4.0}]
    )
    # Instance k = 1 is the new instance: everything runs.
    assert tuner._split_cached(2, 1, [elite, new], ProblemA, 5) == ({}, [0, 1])
    # Afterwards the elite reuses its result on a seen (instance, seed).
    assert tuner._split_cached(2, 2, [elite, new], ProblemA, 5) == ({0: 4.0}, [1])
    # A different seed is a different instance.
    assert tuner._split_cached(2, 2, [elite, new], ProblemA, 6) == ({}, [0, 1])


def test_evaluate_serial_returns_job_order(monkeypatch):
    tuner = make_tuner()
    monkeypatch.setattr(
        tuner, "_run_solver", lambda c, instance, seed: (c["x"] * seed, {})
    )
    jobs = [({"x": float(i)}, ProblemA, 10) for i in range(5)]
    assert [result for result, _ in tuner._evaluate(jobs)] == [0, 10, 20, 30, 40]


# --- tuning time ----------------------------------------------------------------


class _FakeSolver:
    pass


class _FakeConfig:
    identifier = "fake"

    def with_values(self, values):
        return self

    def to_dict(self):
        return {}


RUN_TIMINGS = {"optimizer": {"iterate": {"wall": 1.0, "cpu": 0.5}}, "adapter": {}}


def test_tuning_times_sum_solver_runs_and_tuner_work(monkeypatch, tmp_path):
    tuner = make_tuner(B=120, results_directory=tmp_path)
    executed = []

    def fake_run(c, instance, seed):
        executed.append((c[CONFIG_ID], instance, seed))
        return c["x"] + (seed % 7) / 100.0, RUN_TIMINGS

    monkeypatch.setattr(tuner, "_run_solver", fake_run)
    tuner.tune(
        _FakeSolver, _FakeSolver, None, _FakeConfig(), ParameterSpace([CONTINUOUS])
    )

    times = tuner.tuning_times
    races = times["races"]
    assert [race["race"] for race in races] == list(range(1, len(races) + 1))
    assert len(races) >= 2
    for race in races:
        assert race["tuner"]["wall"] >= 0.0
        for clock in ("wall", "cpu"):
            assert race["total"][clock] == pytest.approx(
                race["solver"][clock] + race["tuner"][clock]
            )

    # Only executed runs count; cached elite results add nothing.
    solver_wall = sum(race["solver"]["wall"] for race in races)
    assert solver_wall == pytest.approx(len(executed) * 1.0)
    assert times["total"]["solver"]["cpu"] == pytest.approx(len(executed) * 0.5)
    for part in ("solver", "tuner", "total"):
        assert times["total"][part]["wall"] == pytest.approx(
            sum(race[part]["wall"] for race in races)
        )

    saved = json.loads((tmp_path / "fake" / "tuning_times.json").read_text())
    assert saved == times
    log = (tmp_path / "fake" / "logs" / "tuning.log").read_text()
    assert "Race 1 time: solver wall=" in log
    assert "Tuning time: solver wall=" in log


def test_tuner_time_excludes_logging(monkeypatch, tmp_path):
    tuner = make_tuner(B=60, results_directory=tmp_path)
    monkeypatch.setattr(
        tuner, "_run_solver", lambda c, instance, seed: (c["x"], RUN_TIMINGS)
    )

    def slow_write(self, name, *messages):
        time.sleep(0.001)

    monkeypatch.setattr(_TuningLogger, "_write", slow_write)
    tuner.tune(
        _FakeSolver, _FakeSolver, None, _FakeConfig(), ParameterSpace([CONTINUOUS])
    )
    # Every race logs far more than 50 lines (> 50 ms) but does little else.
    for race in tuner.tuning_times["races"]:
        assert race["tuner"]["wall"] < 0.05


# --- sampling model -------------------------------------------------------------


def make_model(n_configs=40, seed=0, digits=4):
    model = IFRaceSamplingModel(SPACE, rng=np.random.default_rng(seed))
    model._digits = digits
    return model, model.sample_initial(n_configs)


def test_initial_configurations_are_in_bounds_with_unique_ids():
    model, configurations = make_model()
    for configuration in configurations:
        assert_in_space(configuration)
    assert len({c[CONFIG_ID] for c in configurations}) == len(configurations)


def test_initial_sigma_is_half_the_range_in_log_space_for_log_parameters():
    model, configurations = make_model(n_configs=4)
    sigmas = model.sigmas(configurations[0][CONFIG_ID])
    assert sigmas["x"] == 0.5
    assert sigmas["n"] == 1.5
    assert sigmas["eta"] == pytest.approx((np.log(0.2) - np.log(1e-4)) / 2)
    probabilities = model.categorical_probabilities(configurations[0][CONFIG_ID])
    np.testing.assert_allclose(probabilities["kind"], [1 / 3] * 3)


def test_update_shrinks_sigma_and_moves_categorical_mass():
    model, configurations = make_model(n_configs=4)
    elites = configurations[:2]
    model.update(elites, np.array([2 / 3, 1 / 3]), race_index=1, N_new=8, n_races=4)

    elite = elites[0]
    shrink = (1 / 8) ** (1 / 5)
    assert model.sigmas(elite[CONFIG_ID])["x"] == pytest.approx(0.5 * shrink)
    p = model.categorical_probabilities(elite[CONFIG_ID])["kind"]
    expected = np.full(3, 1 / 3) * (1 - 1 / 4)
    expected[("a", "b", "c").index(elite["kind"])] += 1 / 4
    expected = np.minimum(expected, 0.2 ** (1 / 5))
    np.testing.assert_allclose(p, expected / expected.sum())
    assert model.categorical_update_weight == 1 / 4


def test_categorical_weight_uses_current_number_of_races(monkeypatch, tmp_path):
    # L = 2 is extended several times; the eq. 3 weight must stay below 1 and the
    # categorical distributions must stay valid (no negative probabilities).
    tuner = make_tuner(B=400, L=2, results_directory=tmp_path)
    monkeypatch.setattr(
        tuner,
        "_run_solver",
        lambda c, instance, seed: (c["x"] + ("abc".index(c["kind"])), RUN_TIMINGS),
    )
    calls = []
    original_update = IFRaceSamplingModel.update

    def spy(model, elites, elite_weights, race_index, N_new, n_races):
        original_update(model, elites, elite_weights, race_index, N_new, n_races)
        calls.append((race_index, n_races, model.categorical_update_weight))
        for elite in elites:
            p = model.categorical_probabilities(elite[CONFIG_ID])["kind"]
            assert np.all(p >= 0) and p.sum() == pytest.approx(1.0)

    monkeypatch.setattr(IFRaceSamplingModel, "update", spy)
    tuner.tune(
        _FakeSolver,
        _FakeSolver,
        None,
        _FakeConfig(),
        ParameterSpace([CONTINUOUS, CATEGORICAL]),
    )

    assert tuner.L > 3
    assert max(race_index for race_index, _, _ in calls) >= 2
    for race_index, n_races, weight in calls:
        assert n_races >= race_index + 1
        assert weight == race_index / n_races < 1


def test_sampled_children_are_in_bounds_and_inherit_distributions():
    model, configurations = make_model(n_configs=4)
    elites = configurations[:2]
    model.update(elites, np.array([2 / 3, 1 / 3]), race_index=1, N_new=30, n_races=4)
    children = model.sample(30)

    assert len(children) == len(model.parent_indices) == 30
    for child, parent in zip(children, model.parent_indices, strict=True):
        assert_in_space(child)
        parent_id = elites[parent][CONFIG_ID]
        assert model.sigmas(child[CONFIG_ID]) == model.sigmas(parent_id)


SMALL_LOG_INTEGER = Parameter(
    "k", ParameterType.INTEGER, lower=1, upper=3, log_scale_tuning=True
)
# Each value v gets the log-space share of its cell [v, v + 1) in [1, 4).
LOG_INTEGER_SHARES = [np.log((v + 1) / v) / np.log(4) for v in (1, 2, 3)]


def value_shares(values):
    return [values.count(v) / len(values) for v in (1, 2, 3)]


def test_initial_log_scale_integers_give_each_value_its_log_cell():
    model = IFRaceSamplingModel(
        ParameterSpace([SMALL_LOG_INTEGER]), rng=np.random.default_rng(0)
    )
    values = [c["k"] for c in model.sample_initial(1024)]
    np.testing.assert_allclose(value_shares(values), LOG_INTEGER_SHARES, atol=0.01)


def test_sampled_log_scale_integers_reach_both_bounds_without_bias():
    # With a very wide sigma the truncated normal is nearly uniform in log space,
    # so each value should get its log-cell share (rounding would give 3 only
    # about 17% instead of about 21%).
    model = IFRaceSamplingModel(
        ParameterSpace([SMALL_LOG_INTEGER]), rng=np.random.default_rng(0)
    )
    elite = model.sample_initial(1)[0] | {"k": 2}
    model._sigmas[elite[CONFIG_ID]]["k"] = 1e6
    values = [model._sample_value(SMALL_LOG_INTEGER, elite) for _ in range(20000)]
    assert set(values) == {1, 2, 3}
    np.testing.assert_allclose(value_shares(values), LOG_INTEGER_SHARES, atol=0.015)


@pytest.mark.parametrize(
    "param, value, expected",
    [
        (CONTINUOUS, 0.734567, 0.7346),
        (CONTINUOUS, 0.12341, 0.1234),
        (LOG_CONTINUOUS, 0.1234567, 0.1235),
        (LOG_CONTINUOUS, 0.000149237, 0.0001492),
        (LOG_CONTINUOUS, 0.000151, 0.000151),
    ],
)
def test_round_real_uses_decimals_or_significant_figures(param, value, expected):
    assert _round_real(param, value, 4) == expected


def test_round_real_stays_within_bounds_with_more_digits():
    param = Parameter("p", ParameterType.CONTINUOUS, lower=0.123446, upper=0.123456)
    # 0.123448 -> 0.1234 < lower; 0.123455 -> 0.1235 > upper.
    assert _round_real(param, 0.123448, 4) == 0.123446
    assert _round_real(param, 0.123455, 4) == 0.123456


def test_sampled_reals_are_rounded():
    model, configurations = make_model(n_configs=8)
    model.update(
        configurations[:2], np.array([2 / 3, 1 / 3]), race_index=1, N_new=20, n_races=4
    )
    for c in configurations + model.sample(20):
        assert c["x"] == round(c["x"], 4)
        assert c["eta"] == float(f"{c['eta']:.4g}")


def test_state_accessors_return_copies():
    model, configurations = make_model(n_configs=2)
    config_id = configurations[0][CONFIG_ID]
    model.sigmas(config_id)["x"] = -1.0
    model.categorical_probabilities(config_id)["kind"][:] = 0.0
    assert model.sigmas(config_id)["x"] == 0.5
    assert model.categorical_probabilities(config_id)["kind"].sum() == 1.0


def test_distance_ignores_differences_below_threshold():
    model, _ = make_model(n_configs=1)
    base = {"x": 0.5, "eta": 0.01, "n": 3, "m": 10, "kind": "a"}
    assert model._distance(base, dict(base, x=0.50001)) == 0
    assert model._distance(base, dict(base, x=0.75)) == pytest.approx(0.25)
    assert model._distance(base, dict(base, kind="b")) == 1.0
    # log scale: 0.01 vs 0.02 is log(2) over the log range
    log_range = np.log(0.2) - np.log(1e-4)
    assert model._distance(base, dict(base, eta=0.02)) == pytest.approx(
        np.log(2) / log_range
    )


def test_soft_restart_fires_on_duplicates_and_resets_parents():
    # digits = 0: every numerical difference counts as zero, so any two candidates
    # with the same categorical value are "converged".
    model, configurations = make_model(n_configs=2, digits=0)
    elites = configurations[:1]
    model.update(elites, np.array([1.0]), race_index=1, N_new=10, n_races=4)
    sigma_before = model.sigmas(elites[0][CONFIG_ID])["x"]

    model.sample(10)

    assert model.soft_restarted_elites == (0,)
    N_param = 5
    expected = min(sigma_before * 10 ** (2 / N_param), 0.5 * (1 / 10) ** (1 / N_param))
    assert model.sigmas(elites[0][CONFIG_ID])["x"] == pytest.approx(expected)


def test_child_matching_another_elite_restarts_only_its_parent():
    model, configurations = make_model(n_configs=2)
    elites = configurations[:2]
    model.update(elites, np.array([2 / 3, 1 / 3]), race_index=1, N_new=1, n_races=4)
    # The only child (parent: elite 0) is an exact copy of elite 1.
    child = {name: value for name, value in elites[1].items() if name != CONFIG_ID}
    assert model._find_converged_elites([child], [0]) == {0}


def test_no_soft_restart_when_candidates_differ():
    model, configurations = make_model(n_configs=4)
    model.update(
        configurations[:2], np.array([2 / 3, 1 / 3]), race_index=1, N_new=3, n_races=4
    )
    model.sample(3)
    assert model.soft_restarted_elites == ()


def test_sample_without_elites_raises():
    model, _ = make_model(n_configs=1)
    with pytest.raises(ValueError):
        model.sample(3)
