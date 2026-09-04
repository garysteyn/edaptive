import numpy as np
from edaptive.core.optimizer import BaseOptimizer
from edaptive.const import max_float
from edaptive.utils.timing import timed

class PSO(BaseOptimizer):
    def __init__(self, problem, rng, n_s=30, w=0.7298, c_1=1.49618, c_2=1.49618):
        super().__init__()
        self.initialize(problem=problem, rng=rng, n_s=n_s, w=w, c_1=c_1, c_2=c_2)
        self.init_history()

    def get_required_data(self, names):
        return super().get_required_data(names)

    @timed
    def initialize(self, problem, rng, n_s, w, c_1, c_2):
        self.rng = rng
        
        self.problem = problem
        self.n_x = problem.dimensions
        self.problem_lb = problem.lower
        self.problem_ub = problem.upper

        self._CP = np.column_stack((w, c_1, c_2))

        self.n_s = n_s

        self.X = self.rng.uniform(
            self.problem_lb,
            self.problem_ub,
            size=[self.n_s, self.n_x]
        )

        self.Y = self.X.copy()

        self.f_X = np.ones(self.n_s) * np.inf
        self.f_Y = self.f_X.copy()

        self.V = np.zeros(shape=[self.n_s, self.n_x])
        
        self.Y_hat = None
        self.f_Y_hat = np.inf
    
    def init_history(self):
        self.history["velocities"] = []
        self.history["velocities"].append(self.V.copy())
        
        self.history["diversity"] = []
        self.history["diversity"].append(swarm_diversity(self.X))

        self.history["prop_feasible"] = []
        self.history["prop_feasible"].append(1.0)

        self.history["prop_stable"] = []
        self.history["prop_stable"].append(
            stable_particle_proportion(
                w=self._CP[:, 0],
                c_1=self._CP[:, 1],
                c_2=self._CP[:, 2]
            )
        )

    def step(self, t):
        self.t = t
        self.iterate()
        self.update_history()

    @timed
    def iterate(self):
        for i in range(self.n_s):
            self.f_X[i] = f_X_i = self.problem(self.X[i])
            if np.any((self.X[i] < self.problem.lower) | (self.X[i] > self.problem.upper)):
                continue
            
            if f_X_i < self.f_Y[i]:
                self.Y[i] = self.X[i].copy()
                self.f_Y[i] = f_X_i

            if f_X_i < self.f_Y_hat:
                self.Y_hat = self.X[i].copy()
                self.f_Y_hat = f_X_i

        r_1 = self.rng.uniform(size=(self.n_s, self.n_x))
        r_2 = self.rng.uniform(size=(self.n_s, self.n_x))

        w = self._CP[:, 0, None]
        c_1 = self._CP[:, 1, None]
        c_2 = self._CP[:, 2, None]

        V = (
            w * self.V
            + c_1 * r_1 * (self.Y - self.X)
            + c_2 * r_2 * (self.Y_hat - self.X)
        )
        self.V = np.nan_to_num(V, posinf=max_float, neginf=-max_float)

        self.X += self.V
        self.X = np.nan_to_num(self.X, posinf=max_float, neginf=-max_float)

    def update_history(self):
        feasible = np.all((self.X >= self.problem.lower) & (self.X <= self.problem.upper), axis=1)
        self.history["f_best"].append(self.f_Y_hat)
        self.history["prop_feasible"].append(
            feasible.sum() / self.n_s
        )
        self.history["prop_stable"].append(
            stable_particle_proportion(
                w=self._CP[:, 0],
                c_1=self._CP[:, 1],
                c_2=self._CP[:, 2]
            )
        )

def swarm_diversity(positions: np.ndarray) -> float:
    center = np.mean(positions, axis=0)
    distances = np.linalg.norm(
        positions - center,
        axis=1
    )
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