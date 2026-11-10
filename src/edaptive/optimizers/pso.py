import numpy as np

from edaptive.const import max_float
from edaptive.core.optimizer import BaseOptimizer
from edaptive.utils.timing import timed


class PSO(BaseOptimizer):
    """Global-best PSO with per-particle CPs.

    The initial positions are evaluated in ``initialize``. Each ``iterate`` then
    moves the swarm and evaluates the new positions, so after ``step(t)``, ``f_X``
    is the fitness of ``X`` and ``X`` was produced by the current
    ``adaptable_CPs``. A run of ``T`` steps evaluates ``(T + 1) * n_s`` positions.

    If ``V_max_fraction`` is given, each velocity component is clamped to
    ``[-V_max, V_max]`` with ``V_max = V_max_fraction * (upper - lower)`` per
    dimension; with ``None`` (the default) velocities are not clamped.
    """

    adaptable_CP_names = ("w", "c_1", "c_2")

    @timed
    def initialize(self, problem, rng, n_s, w, c_1, c_2, V_max_fraction=None):
        self.rng = rng

        self.problem = problem
        self.n_x = problem.dimensions
        self.problem_lb = problem.lower
        self.problem_ub = problem.upper

        if V_max_fraction is None:
            self.V_max = None
        else:
            if V_max_fraction <= 0:
                raise ValueError(f"V_max_fraction must be > 0, got {V_max_fraction}")
            self.V_max = V_max_fraction * (self.problem_ub - self.problem_lb)

        # Each CP may be a scalar (same for all particles) or a length-n_s array.
        self._adaptable_CPs = np.column_stack(
            [np.broadcast_to(np.asarray(cp, dtype=float), n_s) for cp in (w, c_1, c_2)]
        )

        self.n_s = n_s

        self.X = self.rng.uniform(
            self.problem_lb, self.problem_ub, size=[self.n_s, self.n_x]
        )

        self.V = np.zeros(shape=[self.n_s, self.n_x])

        self.Y = self.X.copy()
        self.f_Y = np.full(self.n_s, np.inf)

        self.Y_hat = None
        self.f_Y_hat = np.inf

        # No step has been taken yet.
        self.t = None

        self.f_X = self.problem(self.X)
        self._update_bests()

    def init_history(self):
        self.history["f_best"].append(self.f_Y_hat)

        self.history["velocities"] = []
        self.history["velocities"].append(self.V.copy())

        self.history["diversity"] = []
        self.history["diversity"].append(swarm_diversity(self.X))

        self.history["prop_feasible"] = []
        self.history["prop_feasible"].append(1.0)

        self.history["prop_stable"] = []
        self.history["prop_stable"].append(
            stable_particle_proportion(
                w=self._adaptable_CPs[:, 0],
                c_1=self._adaptable_CPs[:, 1],
                c_2=self._adaptable_CPs[:, 2],
            )
        )

    def step(self, t):
        self.t = t
        self.iterate()
        self.update_history()

    def _update_bests(self):
        """Update personal and global bests from ``X``/``f_X``. Only particles
        inside the problem bounds can become a best."""
        valid = np.all(
            (self.X >= self.problem.lower) & (self.X <= self.problem.upper), axis=1
        )

        # Particles that improve their personal best
        improved = valid & (self.f_X < self.f_Y)

        self.Y[improved] = self.X[improved]
        self.f_Y[improved] = self.f_X[improved]

        # Particle that improves global best
        if np.any(valid):
            valid_indices = np.flatnonzero(valid)
            best_idx = valid_indices[np.argmin(self.f_X[valid])]

            if self.f_X[best_idx] < self.f_Y_hat:
                self.Y_hat = self.X[best_idx].copy()
                self.f_Y_hat = self.f_X[best_idx]

    @timed
    def iterate(self):
        # Generate random coefficients
        r_1 = self.rng.uniform(size=(self.n_s, self.n_x))
        r_2 = self.rng.uniform(size=(self.n_s, self.n_x))

        w = self._adaptable_CPs[:, 0, None]
        c_1 = self._adaptable_CPs[:, 1, None]
        c_2 = self._adaptable_CPs[:, 2, None]

        # Update velocities
        V = (
            w * self.V
            + c_1 * r_1 * (self.Y - self.X)
            + c_2 * r_2 * (self.Y_hat - self.X)
        )

        self.V = np.nan_to_num(V, posinf=max_float, neginf=-max_float)

        # Velocity clamping
        if self.V_max is not None:
            self.V = np.clip(self.V, -self.V_max, self.V_max)

        # Update positions
        self.X += self.V

        self.X = np.nan_to_num(self.X, posinf=max_float, neginf=-max_float)

        # Evaluate the new positions
        self.f_X = self.problem(self.X)
        self._update_bests()

    def update_history(self):
        feasible = np.all(
            (self.X >= self.problem.lower) & (self.X <= self.problem.upper), axis=1
        )
        self.history["f_best"].append(self.f_Y_hat)
        self.history["prop_feasible"].append(feasible.sum() / self.n_s)
        self.history["prop_stable"].append(
            stable_particle_proportion(
                w=self._adaptable_CPs[:, 0],
                c_1=self._adaptable_CPs[:, 1],
                c_2=self._adaptable_CPs[:, 2],
            )
        )


def swarm_diversity(positions: np.ndarray) -> float:
    center = np.mean(positions, axis=0)
    distances = np.linalg.norm(positions - center, axis=1)
    return np.mean(distances)


def stable_particle_proportion(
    w: np.ndarray,
    c_1: np.ndarray,
    c_2: np.ndarray,
) -> float:
    w = np.asarray(w)
    c_1 = np.asarray(c_1)
    c_2 = np.asarray(c_2)

    if not (np.all(-1 <= w) and np.all(w <= 1)):
        raise ValueError("All inertia weights must satisfy -1 <= w <= 1.")

    stability_limit = 24 * (1 - w**2) / (7 - 5 * w)

    stable = (c_1 + c_2) < stability_limit

    return np.mean(stable)
