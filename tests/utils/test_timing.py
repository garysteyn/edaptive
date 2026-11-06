"""``@timed`` accumulates wall-clock and CPU time; ``core_time`` sums them."""

import pytest

from edaptive.utils.timing import core_time, timed


class Busy:
    def __init__(self):
        self._timings = {}

    @timed
    def work(self, n):
        return sum(i * i for i in range(n))


def test_timed_returns_result_and_records_both_clocks():
    obj = Busy()
    assert obj.work(10) == sum(i * i for i in range(10))
    assert set(obj._timings) == {"work"}
    assert set(obj._timings["work"]) == {"wall", "cpu"}
    assert obj._timings["work"]["wall"] > 0
    assert obj._timings["work"]["cpu"] >= 0


def test_timed_accumulates_over_calls():
    obj = Busy()
    obj.work(100_000)
    first = dict(obj._timings["work"])
    obj.work(100_000)
    assert obj._timings["work"]["wall"] > first["wall"]
    assert obj._timings["work"]["cpu"] >= first["cpu"]


TIMINGS = {
    "optimizer": {
        "initialize": {"wall": 1.0, "cpu": 0.5},
        "iterate": {"wall": 12.0, "cpu": 10.5},
    },
    "adapter": {"sample": {"wall": 3.0, "cpu": 2.5}},
    "solver": {"_set_adapted_CPs": {"wall": 0.25, "cpu": 0.125}},
}


def test_core_time_sums_all_components():
    assert core_time(TIMINGS) == pytest.approx({"wall": 16.25, "cpu": 13.625})


def test_core_time_of_empty_timings_is_zero():
    assert core_time({"optimizer": {}, "adapter": {}, "solver": {}}) == {
        "wall": 0.0,
        "cpu": 0.0,
    }
