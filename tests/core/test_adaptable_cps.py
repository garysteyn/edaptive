"""Name-based mapping of adaptable CPs between optimizers and adapters."""

import numpy as np
import pytest

from edaptive.adapters.uniform_random import UniformRandomAdapter
from edaptive.core.parameters import Parameter, ParameterSpace, ParameterType
from edaptive.optimizers.pso import PSO
from edaptive.problems.ela_benchmark_suite import Rosenbrock


def make_pso(n_s=4):
    return PSO(
        problem=Rosenbrock(dimensions=2),
        rng=np.random.default_rng(0),
        n_s=n_s,
        w=np.full(n_s, 0.1),
        c_1=np.full(n_s, 1.0),
        c_2=np.full(n_s, 2.0),
    )


def CP_space(*names):
    upper = {"w": 1.0, "c_1": 4.0, "c_2": 4.0}
    return ParameterSpace(
        [
            Parameter(
                name=name, type=ParameterType.CONTINUOUS, lower=0.0, upper=upper[name]
            )
            for name in names
        ]
    )


def test_pso_adaptable_CPs_layout():
    pso = make_pso()
    assert pso.adaptable_CP_names == ("w", "c_1", "c_2")
    np.testing.assert_array_equal(pso.adaptable_CPs, [[0.1, 1.0, 2.0]] * 4)


def test_column_indices_follow_names():
    pso = make_pso()
    assert pso.column_indices(["c_2", "w"]) == [2, 0]


def test_column_indices_rejects_unknown_name():
    with pytest.raises(ValueError, match="no adaptable CPs \\['n_s'\\]"):
        make_pso().column_indices(["n_s"])


def test_set_adaptable_CPs_writes_only_named_columns():
    pso = make_pso()
    before = pso.adaptable_CPs
    pso.set_adaptable_CPs(["c_2", "w"], np.column_stack([np.full(4, 3.5), np.zeros(4)]))
    np.testing.assert_array_equal(pso.adaptable_CPs, [[0.0, 1.0, 3.5]] * 4)
    # The previous array is not modified in place.
    np.testing.assert_array_equal(before, [[0.1, 1.0, 2.0]] * 4)


def test_adapter_selects_own_columns_in_CP_space_order():
    adapter = UniformRandomAdapter(
        CP_space=CP_space("c_2", "w"), rng=np.random.default_rng(0)
    )
    assert adapter.CP_names == ("c_2", "w")
    adaptable_CPs = np.array([[0.1, 1.0, 2.0], [0.2, 1.5, 2.5]])
    np.testing.assert_array_equal(
        adapter.select_own_CPs(adaptable_CPs, ("w", "c_1", "c_2")),
        [[2.0, 0.1], [2.5, 0.2]],
    )
