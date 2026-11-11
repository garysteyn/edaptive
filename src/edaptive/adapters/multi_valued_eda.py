"""Shared model of the multi-valued (r-valued) univariate EDAs of Ben Jedidia,
Doerr & Krejca (2023), "Estimation-of-Distribution Algorithms for Multi-Valued
Decision Variables", GECCO '23.

Each adapted CP is one r-valued decision variable. Its r values depend on its type:
an evenly spaced grid over the bounds (endpoints included) for a continuous CP,
every integer in the bounds for an integer CP, and the classes for a categorical
CP. As in the paper, r is common to all CPs. The model is the n x r
frequency matrix ``p`` (n = number of adapted CPs, rows sum to 1), initialised to
the uniform distribution 1/r and, after every update, restricted to the borders
[1/((r-1)n), 1 - 1/n] with eq. (1) of the paper.
"""

from abc import abstractmethod

import numpy as np

from edaptive.adapters.elite_weighting import WEIGHTED_SCHEMES, elite_weights
from edaptive.const import max_float
from edaptive.core.adapter import EDAAdapter
from edaptive.core.parameters import ParameterType
from edaptive.utils.timing import timed


class MultiValuedEDA(EDAAdapter):
    """Base of r-UMDA, r-PBIL and r-cGA.

    The paper ranks samples by fitness. Here a sample is a particle's CP vector,
    and ``check_update`` ranks only particles inside the problem bounds, every
    ``update_freq`` iterations, by the score that ``elite_signal`` sets:

    - ``"fitness"``: the current fitness;
    - ``"improvement"``: the fitness change since the last update, i.e. over the
      window in which the current CPs were in use. The window starts at the
      initial population (``start``) and restarts whenever an update is returned,
      because the solver resamples the CPs then.

    Lower scores are better (minimisation). Subclasses choose the samples an
    update uses (``select``) and the update rule (``updated_frequencies``).

    ``sample`` returns a float array, unless a categorical CP has a non-numeric
    class; then it returns an object array, which only an optimizer whose
    ``adaptable_CPs`` can hold such values accepts.
    """

    elite_signals = ("fitness", "improvement")

    # Minimum number of feasible particles an update needs. Subclasses set it.
    min_feasible = 2

    def required_data(self):
        return [
            "problem_lb",
            "problem_ub",
            "t",
            "X",
            "f_X",
            "adaptable_CPs",
            "adaptable_CP_names",
        ]

    def initialize_model(self, n_values, update_freq, elite_signal, rng):
        """Set up the values, the uniform frequency matrix and the borders.

        ``n_values`` is the paper's r, the number of values of every CP: the grid
        size of continuous CPs, which integer CPs (``upper - lower + 1`` values)
        and categorical CPs (their classes) must match.
        """
        if elite_signal not in self.elite_signals:
            raise ValueError(
                f"elite_signal must be one of {self.elite_signals}, "
                f"got {elite_signal!r}"
            )

        n_values = int(n_values)
        n = self.n_CPs

        if n_values < 2:
            raise ValueError(f"n_values must be at least 2, got {n_values}")

        # The borders [1/((r-1)n), 1 - 1/n] are empty for n = 1, and collapse to
        # a single point (every frequency fixed at 1/2) for n = r = 2.
        if n < 2:
            raise ValueError(
                f"{type(self).__name__} needs at least 2 adapted CPs, got {n}"
            )
        if n == 2 and n_values == 2:
            raise ValueError("n_values must be at least 3 when 2 CPs are adapted")

        self.rng = rng
        self.n_values = n_values
        self.update_freq = update_freq
        self.elite_signal = elite_signal

        rows = [self._cp_values(parameter, n_values) for parameter in self.CP_space]
        for parameter, row in zip(self.CP_space, rows, strict=True):
            if len(row) != n_values:
                raise ValueError(
                    f"CP {parameter.name!r} has {len(row)} values, but every CP "
                    f"needs n_values = {n_values}"
                )

        # values[i, j] is value j of CP i.
        numeric = all(
            isinstance(value, (int, float, np.number)) and not isinstance(value, bool)
            for row in rows
            for value in row
        )
        self.values = np.array(rows, dtype=float if numeric else object)

        self.lower_border = 1 / ((n_values - 1) * n)
        self.upper_border = 1 - 1 / n

        self.p = np.full((n, n_values), 1 / n_values)

        # Fitness at the start of the current window ("improvement" only).
        self.f_X_window_start = None

    @staticmethod
    def _cp_values(parameter, n_values):
        """The values of one CP, by its type."""
        if parameter.type == ParameterType.CATEGORICAL:
            return list(parameter.classes)
        if parameter.type == ParameterType.INTEGER:
            return list(np.arange(parameter.lower, parameter.upper + 1, dtype=float))
        return list(np.linspace(parameter.lower, parameter.upper, n_values))

    def restrict(self, p):
        """Restrict each row of ``p`` to the borders [a, b], keeping it summing to
        1 (eq. (1)): clamp to [a, b], then scale each frequency's share above a so
        that the shares sum to 1 - a*r."""
        a = self.lower_border
        excess = np.clip(p, a, self.upper_border) - a
        return excess * (1 - a * self.n_values) / excess.sum(axis=1, keepdims=True) + a

    def encode(self, CPs):
        """Value indices of ``CPs`` (shape ``(N, n_CPs)``): the nearest value of a
        continuous or integer CP, the equal class of a categorical CP."""
        indices = np.empty(CPs.shape, dtype=int)
        for i, parameter in enumerate(self.CP_space):
            column = CPs[:, i]
            if parameter.type == ParameterType.CATEGORICAL:
                matches = column[:, None] == self.values[i][None, :]
                if not np.all(matches.any(axis=1)):
                    raise ValueError(
                        f"CP {parameter.name!r} has values that are not among its "
                        f"classes {parameter.classes}"
                    )
                indices[:, i] = np.argmax(matches, axis=1)
            else:
                distances = np.abs(
                    column.astype(float)[:, None] - self.values[i].astype(float)
                )
                indices[:, i] = np.argmin(distances, axis=1)
        return indices

    def relative_frequencies(self, indices):
        """Relative frequency of each value per CP among the rows of ``indices``,
        shape ``(n_CPs, n_values)``."""
        return np.mean(indices[:, :, None] == np.arange(self.n_values), axis=0)

    def elite_frequencies(self, elites):
        """Weighted frequency of each value per CP among ``elites`` (best first):
        ``sum_k w_k * 1{x_i^(k) = j}``, with the weights of ``elite_weighting``
        (as in BetaEDA). Uniform weights give the relative frequencies, the
        paper's update."""
        if self.elite_weighting not in WEIGHTED_SCHEMES:
            return self.relative_frequencies(elites)
        weights = elite_weights(len(elites), self.elite_weighting)
        one_hot = elites[:, :, None] == np.arange(self.n_values)
        return np.sum(weights[:, None, None] * one_hot, axis=0)

    def best(self, scores, indices, k):
        """The ``k`` rows of ``indices`` with the lowest scores, ties broken
        uniformly at random."""
        permutation = self.rng.permutation(len(scores))
        order = permutation[np.argsort(scores[permutation], kind="stable")]
        return indices[order[:k]]

    @timed
    def start(self, f_X, **data):
        if self.elite_signal == "improvement":
            self.f_X_window_start = f_X.copy()

    @timed
    def sample(self, N):
        # Each CP independently takes value j with probability p[i, j] (Sec. 4.2),
        # drawn by inverse CDF: the value index is the number of cumulative
        # frequencies <= u. This is the method of rng.choice, without its
        # per-call overhead. The minimum guards against a last cumulative
        # frequency slightly below 1.
        U = self.rng.random(size=(N, self.n_CPs))
        cdf = np.cumsum(self.p, axis=1)
        indices = np.sum(U[:, :, None] >= cdf[None, :, :], axis=2)
        indices = np.minimum(indices, self.n_values - 1)
        return self.values[np.arange(self.n_CPs), indices]

    @timed
    def check_update(
        self, problem_lb, problem_ub, t, X, f_X, adaptable_CPs, adaptable_CP_names
    ):
        if (t + 1) % self.update_freq != 0:
            return None

        feasible = np.all((X >= problem_lb) & (X <= problem_ub), axis=1)

        if np.count_nonzero(feasible) < self.min_feasible:
            return None

        if self.elite_signal == "improvement":
            # Fitness may be +-max_float, so keep an overflowing difference finite.
            with np.errstate(over="ignore"):
                scores = np.clip(f_X - self.f_X_window_start, -max_float, max_float)
            self.f_X_window_start = f_X.copy()
        else:
            scores = f_X

        indices = self.encode(self.select_own_CPs(adaptable_CPs, adaptable_CP_names))

        return self.select(scores[feasible], indices[feasible])

    @timed
    def update(self, selected):
        self.p = self.restrict(self.updated_frequencies(selected))

    @abstractmethod
    def select(self, scores, indices):
        """The value indices (rows of ``indices``) the update uses, given the
        feasible particles' scores."""

    @abstractmethod
    def updated_frequencies(self, selected):
        """The frequency matrix after the update, before the restriction."""
