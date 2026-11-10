"""PSO-TVAC follows the linear w / c_1 / c_2 schedules of Ratnaweera et al. (2004)."""

import numpy as np
import pytest

from edaptive.core.config import SolverConfig
from edaptive.problems.ela_benchmark_suite import Rosenbrock
from edaptive.solvers.pso.tvac_pso import TVACPSO

N_S = 4
MAX_ITER = 10


def build():
    return TVACPSO(
        problem=Rosenbrock(dimensions=2),
        config=SolverConfig(fixed_CPs={"n_s": N_S}),
        rnd_seed=0,
    )


def expected_CPs(iteration):
    fraction = iteration / MAX_ITER
    return (
        0.9 - 0.5 * fraction,  # w: 0.9 -> 0.4
        2.5 - 2.0 * fraction,  # c_1: 2.5 -> 0.5
        0.5 + 2.0 * fraction,  # c_2: 0.5 -> 2.5
    )


def test_first_step_uses_the_initial_values():
    solver = build()
    np.testing.assert_allclose(
        solver.optimizer.adaptable_CPs, np.tile((0.9, 2.5, 0.5), (N_S, 1))
    )


def test_every_step_uses_the_scheduled_values():
    solver = build()
    solver.max_iterations = MAX_ITER
    for t in range(MAX_ITER):
        np.testing.assert_allclose(
            solver.optimizer.adaptable_CPs, np.tile(expected_CPs(t), (N_S, 1))
        )
        solver.optimizer.step(t)
        solver.adapt(t)


def test_run_ends_on_the_final_values():
    solver = build()
    solver.run(max_iterations=MAX_ITER)
    # The last adapt call sets iter = MAXITR, which no step uses.
    np.testing.assert_allclose(
        solver.optimizer.adaptable_CPs, np.tile((0.4, 0.5, 2.5), (N_S, 1))
    )


@pytest.mark.parametrize("seed", [0, 1])
def test_same_seed_reproduces(seed):
    def run():
        return TVACPSO(
            problem=Rosenbrock(dimensions=2),
            config=SolverConfig(fixed_CPs={"n_s": N_S}),
            rnd_seed=seed,
        ).run(max_iterations=MAX_ITER)[1]["f_best"]

    assert run() == run()
