"""Characterisation tests: pinned seeded results of the current code (R8a).

These pin behaviour, not correctness. A failure means results changed. If that
was intended, regenerate the reference (see ``make_reference.py``) and explain
the change in the commit message; otherwise it's a regression.

Comparisons are exact (bit-for-bit). PSO amplifies last-bit floating-point
differences, so the reference may not match on a machine with different
numpy/scipy/libm builds even when nothing is wrong. It was generated in the
dev container.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from tests.characterisation import cases

REFERENCE = json.loads((Path(__file__).with_name("reference.json")).read_text())


def assert_matches(actual, expected):
    assert actual.keys() == expected.keys()
    for key in expected:
        a, e = actual[key], expected[key]
        if isinstance(e, (list, float, int)) and not isinstance(e, bool):
            np.testing.assert_array_equal(np.asarray(a), np.asarray(e), err_msg=key)
        else:
            assert a == e, key


# --- solvers ----------------------------------------------------------------


@pytest.mark.parametrize("case_id", cases.solver_case_ids())
def test_solver_matches_reference(case_id):
    case_name, problem_name = case_id.split("/")
    actual = cases.run_solver_case(case_name, problem_name)
    assert_matches(actual, REFERENCE["solvers"][case_id])


@pytest.mark.parametrize("case_name", list(cases.SOLVER_CASES))
def test_solver_same_seed_reproduces(case_name):
    first = cases.run_solver_case(case_name, "Rosenbrock", seed=11)
    second = cases.run_solver_case(case_name, "Rosenbrock", seed=11)
    assert_matches(second, first)


@pytest.mark.parametrize("case_name", list(cases.SOLVER_CASES))
def test_solver_different_seed_differs(case_name):
    first = cases.run_solver_case(case_name, "Rosenbrock", seed=11)
    second = cases.run_solver_case(case_name, "Rosenbrock", seed=12)
    assert first["X"] != second["X"]


@pytest.mark.xfail(
    strict=True,
    reason="Known issue 1: BetaEDA's copula sampler uses the global numpy RNG, "
    "so copula runs aren't reproducible from rnd_seed. Remove this marker (and "
    "pin a copula case) once it's fixed.",
)
def test_copula_same_seed_reproduces():
    np.random.seed(1)
    first = cases.run_solver(*cases.COPULA_CASE, cases.Rosenbrock, seed=11)
    np.random.seed(2)
    second = cases.run_solver(*cases.COPULA_CASE, cases.Rosenbrock, seed=11)
    assert_matches(second, first)


# --- elitist I/F-race -------------------------------------------------------


@pytest.mark.filterwarnings("ignore:The balance properties of Sobol:UserWarning")
@pytest.mark.parametrize("case_name", list(cases.TUNING_CASES))
def test_tuning_matches_reference(case_name, tmp_path):
    actual = cases.run_tuning_case(case_name, tmp_path)
    expected = REFERENCE["tuning"][case_name]
    assert actual["L"] == expected["L"]
    assert actual["race_history"] == expected["race_history"]
    assert actual["best_configuration"] == expected["best_configuration"]
    assert (
        tmp_path
        / cases.TUNING_CASES[case_name][2]["identifier"]
        / "best_configuration.pkl"
    ).exists()
