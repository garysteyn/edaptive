"""Shared model of the EMNA adapters (Estimation of Multivariate Normal Algorithm),
as described in Larrañaga (2002), "A Review on Estimation of Distribution
Algorithms", §4.4 (EMNA_global, EMNA_a).

The model is a multivariate normal N(mu, Sigma) over the adapted CPs, each rescaled
to [0, 1] by its bounds ("unit space"), so ``sigma_min`` is a fraction of every
CP's range. Additions to the paper's algorithms:

- samples are drawn from the normal truncated to the CP bounds, by rejection
  sampling;
- a learning rate ``eta`` blends the model toward each update's estimate
  (``eta = 1`` gives the paper's model);
- a minimum standard deviation ``sigma_min``: the eigenvalues of Sigma are kept
  at or above ``sigma_min**2``, so the standard deviation in every direction is at
  least ``sigma_min``;
- rank-based elite weighting (``elite_weighting``, as in BetaEDA).
"""

from abc import abstractmethod

import numpy as np

from edaptive.const import max_float
from edaptive.core.adapter import EDAAdapter
from edaptive.core.parameters import ParameterType
from edaptive.utils.timing import timed


class MultivariateNormalEDA(EDAAdapter):
    """Base of EMNAGlobal and EMNAAdaptive.

    Every ``update_freq`` iterations, ``check_update`` ranks the particles inside
    the problem bounds by the score that ``elite_signal`` sets:

    - ``"fitness"``: the current fitness;
    - ``"improvement"``: the fitness change since the last update, i.e. over the
      window in which the current CPs were in use. The window starts at the
      initial population (``start``) and restarts whenever an update is returned,
      because the solver resamples the CPs then.

    Lower scores are better. Subclasses turn the ranked particles into the
    paper's estimate of the mean and covariance (``estimate``).

    Until the first update, ``sample`` draws uniformly within the CP bounds (the
    paper's random initial population). The initial mean and covariance are
    those of that uniform distribution (0.5 and 1/12 per CP in unit space), the
    starting point of the ``eta`` blend.
    """

    elite_signals = ("fitness", "improvement")

    # Upper limit on rejection-sampling rounds in ``sample``. The mean always lies
    # inside the unit box and the elite variances are at most 0.25, so a round
    # accepts a sizeable fraction of its draws; reaching the limit indicates a bug.
    MAX_REJECTION_ROUNDS = 1000

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

    def initialize_model(
        self,
        eta,
        sigma_min,
        n_elites,
        update_freq,
        elite_weighting,
        elite_signal,
        rng,
    ):
        for parameter in self.CP_space:
            if parameter.type != ParameterType.CONTINUOUS:
                raise ValueError(
                    f"{type(self).__name__} needs continuous CPs, but "
                    f"{parameter.name!r} is {parameter.type.value}"
                )

        if elite_signal not in self.elite_signals:
            raise ValueError(
                f"elite_signal must be one of {self.elite_signals}, "
                f"got {elite_signal!r}"
            )

        if not 0 < eta <= 1:
            raise ValueError(f"eta must lie in (0, 1], got {eta}")

        # 0.5 is the largest standard deviation of a distribution on [0, 1].
        if not 0 < sigma_min <= 0.5:
            raise ValueError(f"sigma_min must lie in (0, 0.5], got {sigma_min}")

        if int(n_elites) < 2:
            raise ValueError(f"n_elites must be at least 2, got {n_elites}")

        self.rng = rng
        self.eta = eta
        self.sigma_min = sigma_min
        self.n_elites = int(n_elites)
        self.update_freq = update_freq
        self.elite_weighting = elite_weighting
        self.elite_signal = elite_signal

        n = self.n_CPs
        self.mu = np.full(n, 0.5)
        self.Sigma = np.eye(n) / 12
        # Sampling factor A with A @ A.T = Sigma, set when the model is fitted.
        self.A = None
        self.fitted = False

        # Fitness at the start of the current window ("improvement" only).
        self.f_X_window_start = None

    @timed
    def start(self, f_X, **data):
        if self.elite_signal == "improvement":
            self.f_X_window_start = f_X.copy()

    @timed
    def sample(self, N):
        if not self.fitted:
            unit_samples = self.rng.random(size=(N, self.n_CPs))
        else:
            unit_samples = self._truncated_normal(N)
        return self.CP_lbs + unit_samples * (self.CP_ubs - self.CP_lbs)

    def _truncated_normal(self, N):
        """``N`` draws from N(mu, Sigma) truncated to the unit box, by rejection."""
        accepted = [np.empty((0, self.n_CPs))]
        remaining = N
        rounds = 0
        while remaining > 0:
            if rounds == self.MAX_REJECTION_ROUNDS:
                raise RuntimeError(
                    f"rejection sampling accepted only {N - remaining} of {N} "
                    f"samples in {self.MAX_REJECTION_ROUNDS} rounds"
                )
            draws = self.rng.standard_normal(size=(remaining, self.n_CPs))
            draws = draws @ self.A.T + self.mu
            inside = draws[np.all((draws >= 0) & (draws <= 1), axis=1)]
            accepted.append(inside)
            remaining -= len(inside)
            rounds += 1
        return np.concatenate(accepted, axis=0)

    @timed
    def check_update(
        self, problem_lb, problem_ub, t, X, f_X, adaptable_CPs, adaptable_CP_names
    ):
        if (t + 1) % self.update_freq != 0:
            return None

        feasible = np.all((X >= problem_lb) & (X <= problem_ub), axis=1)

        if np.count_nonzero(feasible) < 2:
            return None

        # Lower score is better.
        if self.elite_signal == "improvement":
            # Fitness may be +-max_float, so keep an overflowing difference finite.
            with np.errstate(over="ignore"):
                scores = np.clip(f_X - self.f_X_window_start, -max_float, max_float)
            self.f_X_window_start = f_X.copy()
        else:
            scores = f_X

        adapted_CPs = self.select_own_CPs(adaptable_CPs, adaptable_CP_names)
        unit_CPs = (adapted_CPs[feasible] - self.CP_lbs) / (self.CP_ubs - self.CP_lbs)
        scores = scores[feasible]

        order = np.argsort(scores, kind="stable")
        return scores[order], unit_CPs[order]

    @timed
    def update(self, candidates):
        scores, unit_CPs = candidates
        mu_hat, Sigma_hat = self.estimate(scores, unit_CPs)
        self.set_model(mu_hat, Sigma_hat)

    @staticmethod
    def weighted_moments(unit_CPs, weights):
        """Weighted mean and covariance of ``unit_CPs`` (rows). Plain weighted
        moments, without Bessel correction: with uniform weights these are the
        maximum-likelihood estimates of eqs. (3.31)-(3.33)."""
        mu_hat = weights @ unit_CPs
        diff = unit_CPs - mu_hat
        Sigma_hat = (weights[:, None] * diff).T @ diff
        return mu_hat, Sigma_hat

    def set_model(self, mu_hat, Sigma_hat):
        """Blend the model toward ``(mu_hat, Sigma_hat)`` with ``eta``, then raise
        the eigenvalues of Sigma to at least ``sigma_min**2``."""
        self.mu = (1 - self.eta) * self.mu + self.eta * mu_hat
        Sigma = (1 - self.eta) * self.Sigma + self.eta * Sigma_hat
        Sigma = (Sigma + Sigma.T) / 2

        eigenvalues, eigenvectors = np.linalg.eigh(Sigma)
        eigenvalues = np.maximum(eigenvalues, self.sigma_min**2)

        self.Sigma = (eigenvectors * eigenvalues) @ eigenvectors.T
        self.A = eigenvectors * np.sqrt(eigenvalues)
        self.fitted = True

    @abstractmethod
    def estimate(self, scores, unit_CPs):
        """The paper's estimate ``(mu_hat, Sigma_hat)`` in unit space, given the
        feasible particles' scores and unit-space CPs, sorted best first."""
