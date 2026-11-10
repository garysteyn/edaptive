"""Every solver is built with the same constructor and validates its types."""

import pytest

from edaptive.adapters.beta_eda import BetaEDA
from edaptive.adapters.uniform_random import UniformRandomAdapter
from edaptive.core.config import SolverConfig
from edaptive.core.parameters import Parameter, ParameterSpace, ParameterType
from edaptive.optimizers.pso import PSO
from edaptive.problems.ela_benchmark_suite import Rosenbrock
from edaptive.registry import SOLVERS
from edaptive.solvers.basic_solver import BasicSolver
from edaptive.solvers.eda_solver import EDASolver
from edaptive.solvers.pso.sg_pso import StabilityGuidedPSO
from edaptive.solvers.uniform_random_solver import UniformRandomSolver


class OtherOptimizer(PSO):
    pass


ADAPTED = ParameterSpace(
    [
        Parameter(name="w", type=ParameterType.CONTINUOUS, lower=0.0, upper=1.0),
        Parameter(name="c_1", type=ParameterType.CONTINUOUS, lower=0.0, upper=4.0),
        Parameter(name="c_2", type=ParameterType.CONTINUOUS, lower=0.0, upper=4.0),
    ]
)
BETA_EDA_PARAMS = {
    "eta": 0.1,
    "rho_min": 0.02,
    "n_elites": 5,
    "update_freq": 1,
    "copula": None,
    "elite_weighting": None,
}

# solver -> (optimizer_type, adapter_type, config) a caller would pass
CASES = {
    BasicSolver: (
        PSO,
        None,
        SolverConfig(fixed_CPs={"n_s": 6, "w": 0.7, "c_1": 1.4, "c_2": 1.4}),
    ),
    EDASolver: (
        PSO,
        BetaEDA,
        SolverConfig(
            fixed_CPs={"n_s": 6},
            adapted_CP_space=ADAPTED,
            adapter_params=BETA_EDA_PARAMS,
        ),
    ),
    UniformRandomSolver: (
        PSO,
        None,
        SolverConfig(fixed_CPs={"n_s": 6}, adapted_CP_space=ADAPTED),
    ),
    StabilityGuidedPSO: (None, None, SolverConfig(fixed_CPs={"n_s": 6})),
}


def build(solver_type, optimizer_type, adapter_type, config):
    return solver_type(
        optimizer_type,
        adapter_type,
        problem=Rosenbrock(dimensions=2),
        config=config,
        rnd_seed=0,
    )


@pytest.mark.parametrize("solver_type", sorted(set(SOLVERS.values()), key=str))
def test_every_registered_solver_builds_and_runs(solver_type):
    solver = build(solver_type, *CASES[solver_type])
    f_final, history = solver.run(max_iterations=5)
    # One entry for the initial swarm plus one per iteration.
    assert len(history["f_best"]) == 5 + 1
    assert f_final == min(history["f_best"])


@pytest.mark.parametrize(
    ("solver_type", "optimizer_type", "adapter_type", "expected"),
    [
        (StabilityGuidedPSO, PSO, None, (PSO, None)),
        (UniformRandomSolver, PSO, UniformRandomAdapter, (PSO, UniformRandomAdapter)),
        (UniformRandomSolver, PSO, None, (PSO, UniformRandomAdapter)),
    ],
)
def test_hardwired_types_accept_none_or_their_own(
    solver_type, optimizer_type, adapter_type, expected
):
    config = CASES[solver_type][2]
    solver = build(solver_type, optimizer_type, adapter_type, config)
    assert (solver.optimizer_type, solver.adapter_type) == expected


@pytest.mark.parametrize(
    ("solver_type", "optimizer_type", "adapter_type", "message"),
    [
        (StabilityGuidedPSO, OtherOptimizer, None, "always uses PSO"),
        (StabilityGuidedPSO, None, BetaEDA, "takes no adapter"),
        (UniformRandomSolver, PSO, BetaEDA, "always uses UniformRandomAdapter"),
        (BasicSolver, PSO, BetaEDA, "takes no adapter"),
        (EDASolver, PSO, None, "needs an adapter_type"),
        (EDASolver, None, BetaEDA, "needs an optimizer_type"),
    ],
)
def test_rejects_mismatched_types(solver_type, optimizer_type, adapter_type, message):
    config = CASES[solver_type][2]
    with pytest.raises(ValueError, match=message):
        build(solver_type, optimizer_type, adapter_type, config)
