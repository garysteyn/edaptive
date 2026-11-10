"""Every solver reports the timings of its optimizer, adapter and own logic."""

import pytest

from edaptive.problems.ela_benchmark_suite import Rosenbrock
from edaptive.solvers.basic_solver import BasicSolver
from edaptive.solvers.eda_solver import EDASolver
from edaptive.solvers.pso.sg_pso import StabilityGuidedPSO
from edaptive.solvers.uniform_random_solver import UniformRandomSolver
from edaptive.utils.timing import core_time
from tests.solvers.test_construction import CASES

EXPECTED_KEYS = {
    BasicSolver: {
        "optimizer": {"initialize", "iterate"},
        "adapter": set(),
        "solver": set(),
    },
    EDASolver: {
        "optimizer": {"initialize", "iterate"},
        "adapter": {"initialize", "start", "check_update", "update", "sample"},
        "solver": {"_get_required_data", "_set_adapted_CPs"},
    },
    UniformRandomSolver: {
        "optimizer": {"initialize", "iterate"},
        "adapter": {"initialize", "sample"},
        "solver": {"_set_adapted_CPs"},
    },
    StabilityGuidedPSO: {
        "optimizer": {"initialize", "iterate"},
        "adapter": set(),
        "solver": {"sample_adapted_CPs", "_set_adapted_CPs"},
    },
}


def run(solver_type, iterations=5):
    optimizer_type, adapter_type, config = CASES[solver_type]
    solver = solver_type(
        optimizer_type,
        adapter_type,
        problem=Rosenbrock(dimensions=3),
        config=config,
        rnd_seed=0,
    )
    _, history = solver.run(max_iterations=iterations)
    return history["timings"]


@pytest.mark.parametrize("solver_type", list(EXPECTED_KEYS), ids=lambda s: s.__name__)
def test_timings_structure(solver_type):
    timings = run(solver_type)

    assert set(timings) == {"optimizer", "adapter", "solver"}
    for component, methods in EXPECTED_KEYS[solver_type].items():
        assert set(timings[component]) == methods, component
        for times in timings[component].values():
            assert set(times) == {"wall", "cpu"}
            assert times["wall"] >= 0 and times["cpu"] >= 0

    assert core_time(timings)["wall"] > 0
