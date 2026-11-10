"""PSO: evaluation order, histories, feasibility-gated bests and velocity clamping."""

import numpy as np
import pytest

from edaptive.optimizers.pso import PSO
from edaptive.problems.benchmark_problem import BenchmarkProblem
from edaptive.problems.ela_benchmark_suite import Rosenbrock


class Sum(BenchmarkProblem):
    """f(x) = sum(x) on [0, 1]^D: lower is better, so leaving through the lower
    bound gives an infeasible particle with the lowest fitness."""

    def __init__(self, dimensions):
        super().__init__(dimensions)
        self.lower = np.zeros(dimensions)
        self.upper = np.ones(dimensions)

    def _evaluate(self, x):
        return np.sum(x, axis=-1)


def make_pso(problem=None, n_s=6, w=0.7, c_1=1.5, c_2=1.5, seed=0, **optional_CPs):
    return PSO(
        problem=problem if problem is not None else Rosenbrock(dimensions=3),
        rng=np.random.default_rng(seed),
        n_s=n_s,
        w=w,
        c_1=c_1,
        c_2=c_2,
        **optional_CPs,
    )


def test_initial_swarm_is_evaluated():
    pso = make_pso()
    np.testing.assert_array_equal(pso.f_X, pso.problem(pso.X))
    np.testing.assert_array_equal(pso.Y, pso.X)
    np.testing.assert_array_equal(pso.f_Y, pso.f_X)
    assert pso.f_Y_hat == np.min(pso.f_X)
    assert pso.history["f_best"] == [pso.f_Y_hat]


def test_f_X_is_fitness_of_X_after_every_step():
    pso = make_pso()
    for t in range(10):
        pso.step(t)
        np.testing.assert_array_equal(pso.f_X, pso.problem(pso.X))
    assert len(pso.history["f_best"]) == 10 + 1


def test_X_is_produced_by_current_CPs():
    # With c_1 = c_2 = 0 a step is X += w * V, so the move only depends on w.
    pso = make_pso(problem=Sum(dimensions=2), n_s=3, c_1=0.0, c_2=0.0)
    pso.V = np.full((3, 2), 0.1)
    pso.set_adaptable_CPs(["w"], np.array([[0.0], [0.5], [1.0]]))
    X_before = pso.X.copy()
    pso.step(0)
    np.testing.assert_allclose(pso.X - X_before, [[0.0] * 2, [0.05] * 2, [0.1] * 2])


def test_infeasible_particles_never_become_bests():
    pso = make_pso(problem=Sum(dimensions=2), n_s=3, w=1.0, c_1=0.0, c_2=0.0)
    f_Y_hat_before = pso.f_Y_hat
    # Particle 0 leaves through the lower bound (lowest fitness, infeasible);
    # particle 1 moves inside the bounds towards lower fitness.
    pso.X = np.array([[0.1, 0.1], [0.5, 0.5], [0.9, 0.9]])
    pso.f_X = pso.problem(pso.X)
    pso.Y, pso.f_Y = pso.X.copy(), pso.f_X.copy()
    pso.V = np.array([[-1.0, -1.0], [-0.4, -0.4], [0.0, 0.0]])
    pso.step(0)

    assert pso.f_X[0] < pso.f_X[1]
    np.testing.assert_array_equal(pso.Y[0], [0.1, 0.1])
    np.testing.assert_allclose(pso.Y[1], [0.1, 0.1])
    assert pso.f_Y_hat == min(f_Y_hat_before, pso.f_X[1])
    assert pso.f_Y_hat > pso.f_X[0]


def test_velocities_are_clamped_to_V_max():
    # Large c_1/c_2 make the unclamped velocities exceed V_max.
    pso = make_pso(w=1.0, c_1=4.0, c_2=4.0, V_max_fraction=0.1)
    V_max = 0.1 * (pso.problem.upper - pso.problem.lower)
    clamped = False
    for t in range(20):
        pso.step(t)
        assert np.all(np.abs(pso.V) <= V_max)
        clamped |= np.any(np.abs(pso.V) == V_max)
    assert clamped


def test_no_V_max_fraction_means_no_clamping():
    default, explicit = make_pso(), make_pso(V_max_fraction=None)
    for t in range(10):
        default.step(t)
        explicit.step(t)
    np.testing.assert_array_equal(default.X, explicit.X)
    np.testing.assert_array_equal(default.V, explicit.V)


@pytest.mark.parametrize("V_max_fraction", [0.0, -0.5])
def test_non_positive_V_max_fraction_raises(V_max_fraction):
    with pytest.raises(ValueError, match="V_max_fraction"):
        make_pso(V_max_fraction=V_max_fraction)
