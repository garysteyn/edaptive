"""Solvers adapt CPs by name: subsets and reordered "adapted" spaces work."""

import numpy as np
import pytest

from edaptive.adapters.beta_eda import BetaEDA
from edaptive.core.config import SolverConfig
from edaptive.core.parameters import Parameter, ParameterSpace, ParameterType
from edaptive.optimizers.pso import PSO
from edaptive.problems.ela_benchmark_suite import Rosenbrock
from edaptive.solvers.eda_solver import EDASolver
from edaptive.solvers.uniform_random_solver import UniformRandomSolver

UPPER = {"w": 1.0, "c_1": 4.0, "c_2": 4.0}
N_S = 10


def CP_space(*names):
    return ParameterSpace(
        [
            Parameter(
                name=name, type=ParameterType.CONTINUOUS, lower=0.0, upper=UPPER[name]
            )
            for name in names
        ]
    )


ADAPTER_PARAMS = {
    "eta": 0.1,
    "rho_min": 0.02,
    "n_elites": 5,
    "update_freq": 1,
    "copula": None,
    "elite_weighting": "log-rank",
    "elite_signal": "fitness",
}


def build(solver_type, adapted_names, fixed):
    config = SolverConfig(
        fixed_CPs={"n_s": N_S} | fixed,
        adapted_CP_space=CP_space(*adapted_names),
        adapter_params=ADAPTER_PARAMS if solver_type is EDASolver else {},
    )
    adapter_type = BetaEDA if solver_type is EDASolver else None
    return solver_type(
        PSO,
        adapter_type,
        problem=Rosenbrock(dimensions=3),
        config=config,
        rnd_seed=0,
    )


@pytest.mark.parametrize("solver_type", [EDASolver, UniformRandomSolver])
def test_subset_leaves_fixed_CPs_alone(solver_type):
    solver = build(solver_type, ["w"], {"c_1": 1.5, "c_2": 1.5})
    solver.run(max_iterations=30)
    CPs = solver.optimizer.adaptable_CPs
    np.testing.assert_array_equal(CPs[:, 1:], 1.5)
    assert len(np.unique(CPs[:, 0])) > 1
    assert np.all((0 <= CPs[:, 0]) & (CPs[:, 0] <= 1))


@pytest.mark.parametrize("solver_type", [EDASolver, UniformRandomSolver])
def test_reordered_space_maps_columns_by_name(solver_type):
    # w's range is [0, 1] and c's is [0, 4]; a positional mix-up would put
    # c values (up to 4) in the w column.
    solver = build(solver_type, ["c_2", "w", "c_1"], {})
    solver.run(max_iterations=30)
    CPs = solver.optimizer.adaptable_CPs
    assert np.all((0 <= CPs[:, 0]) & (CPs[:, 0] <= 1))
    assert CPs[:, 1:].max() > 1
