import numpy as np
from scipy.stats import beta, norm

from edaptive.adapters.elite_weighting import elite_weights
from edaptive.const import max_float
from edaptive.core.adapter import EDAAdapter
from edaptive.utils.timing import timed


class BetaEDA(EDAAdapter):
    """Beta marginals (optionally joined by a Gaussian copula) fitted to elite CPs.

    Every ``update_freq`` iterations the CPs of the ``n_elites`` best feasible
    particles become the elites. ``elite_signal`` sets what "best" means:

    - ``"fitness"``: the lowest current fitness;
    - ``"improvement"``: the largest fitness decrease since the last update, i.e.
      over the window in which the current CPs were in use. The window starts
      at the initial population (``start``) and restarts whenever elites are
      returned, because the solver resamples the CPs then.
    """

    elite_signals = ("fitness", "improvement")

    # Numerical safeguards (not hyperparameters).
    # Scaled elites are kept in [UNIT_MARGIN, 1 - UNIT_MARGIN], away from the
    # boundaries of beta.cdf/ppf, so that mu * (1 - mu) > 0.
    UNIT_MARGIN = 1e-6
    # Upper clip on the normalised variance rho, so kappa = 1/rho - 1 > 0 and the
    # Beta parameters stay positive.
    RHO_MAX = 1 - 1e-8
    # Diagonal jitter that keeps R + jitter * I positive definite for Cholesky.
    CHOLESKY_JITTER = 1e-10
    # Copula uniforms are kept in [COPULA_MARGIN, 1 - COPULA_MARGIN] so that
    # norm.ppf stays finite.
    COPULA_MARGIN = 1e-12

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

    @timed
    def initialize(
        self,
        eta,
        rho_min,
        n_elites,
        update_freq,
        copula,
        elite_weighting,
        elite_signal,
        rng,
    ):
        if elite_signal not in self.elite_signals:
            raise ValueError(
                f"elite_signal must be one of {self.elite_signals}, "
                f"got {elite_signal!r}"
            )

        if not 0 < rho_min < self.RHO_MAX:
            raise ValueError(f"rho_min must lie in (0, {self.RHO_MAX}), got {rho_min}")

        if int(n_elites) < 2:
            raise ValueError(f"n_elites must be at least 2, got {n_elites}")

        self.rng = rng

        self.copula = copula

        self.a = np.ones(shape=self.n_CPs, dtype=np.float64)
        self.b = np.ones(shape=self.n_CPs, dtype=np.float64)

        if copula == "gaussian":
            self.R = np.eye(N=self.n_CPs)
            self.cholesky_jitter = self.CHOLESKY_JITTER * np.eye(self.n_CPs)

        self.eta = eta
        self.rho_min = rho_min
        self.n_elites = int(n_elites)
        self.update_freq = update_freq

        self.elite_weighting = elite_weighting
        self.elite_signal = elite_signal

        # Fitness at the start of the current window ("improvement" only).
        self.f_X_window_start = None

    @timed
    def start(self, f_X, **data):
        if self.elite_signal == "improvement":
            self.f_X_window_start = f_X.copy()

    @timed
    def sample(self, N):
        if self.copula == "gaussian":
            L = np.linalg.cholesky(self.R + self.cholesky_jitter)
            Z = self.rng.standard_normal(size=(N, self.n_CPs)) @ L.T
            U = norm.cdf(Z)
            samples = beta.ppf(U, self.a, self.b)
        else:
            samples = beta.rvs(
                self.a, self.b, size=(N, self.n_CPs), random_state=self.rng
            )
        return self.CP_lbs + samples * (self.CP_ubs - self.CP_lbs)

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
                score = np.clip(f_X - self.f_X_window_start, -max_float, max_float)
            self.f_X_window_start = f_X.copy()
        else:
            score = f_X

        adapted_CPs = self.select_own_CPs(adaptable_CPs, adaptable_CP_names)

        elite_CPs = self.get_elite_CPs(score[feasible], adapted_CPs[feasible])
        return elite_CPs

    def get_elite_CPs(self, score_feasible, adapted_CPs_feasible):
        """CPs of the (at most) ``n_elites`` particles with the lowest score."""
        return adapted_CPs_feasible[np.argsort(score_feasible)[: self.n_elites]]

    @timed
    def update(self, elite_CPs):
        scaled_CPs = (elite_CPs - self.CP_lbs) / (self.CP_ubs - self.CP_lbs)
        scaled_CPs = np.clip(scaled_CPs, self.UNIT_MARGIN, 1 - self.UNIT_MARGIN)

        # Step 1 (Fit marginals):
        weights = elite_weights(len(scaled_CPs), self.elite_weighting)

        # Plain weighted moments of the elite set (no Bessel correction): the elites
        # are selected by performance, not sampled at random, so the model is fitted
        # to the elite set itself.
        mu = np.average(scaled_CPs, axis=0, weights=weights)
        diff = scaled_CPs - mu
        var = np.sum(weights[:, None] * diff**2, axis=0)

        # Method of moments through the normalised variance rho = var / (mu(1-mu))
        # = 1 / (kappa + 1). rho_min is a floor on it (so kappa <= 1/rho_min - 1),
        # which also covers var = 0 for identical elites.
        rho = np.clip(var / (mu * (1 - mu)), self.rho_min, self.RHO_MAX)
        kappa = 1 / rho - 1

        a_new = mu * kappa
        b_new = (1 - mu) * kappa

        self.a = (1 - self.eta) * self.a + self.eta * a_new
        self.b = (1 - self.eta) * self.b + self.eta * b_new

        if self.copula == "gaussian":
            U = beta.cdf(scaled_CPs, self.a, self.b)

            U = np.clip(U, self.COPULA_MARGIN, 1 - self.COPULA_MARGIN)

            Z = norm.ppf(U)  # Z is approximately multivariate normal

            # Estimate Gaussian copula. A CP on which all elites agree has no
            # defined correlation, so R is then left unchanged.
            if np.all(np.ptp(Z, axis=0) > 0):
                R_new = np.corrcoef(Z.T)
                self.R = (1 - self.eta) * self.R + self.eta * R_new


# class BetaMarginalsEDA(EDAAdapter):
#     def required_data(self):
#         return ["problem_lb", "problem_ub", "t", "X", "f_X", "CP"]

#     def initialize(self, eta, rho_min, n_elite, update_freq, rng):
#         self.rng = rng

#         self.a = np.ones(shape=self.n_x, dtype=np.float64)
#         self.b = np.ones(shape=self.n_x, dtype=np.float64)

#         self.eta = eta
#         self.rho_min = rho_min
#         self.kappa_max = (1 / rho_min) - 1
#         self.n_elite = int(n_elite)
#         self.update_freq = update_freq

#     def sample(self, N):
#         samples = beta.rvs(
#             self.a[None, :],
#             self.b[None, :],
#             size=(N, self.n_x),
#             random_state=self.rng
#         )
#         return self.CP_lb + samples * (self.CP_ub - self.CP_lb)

#     def check_update(self, problem_lb, problem_ub, t, X, f_X, CP):
#         if (t + 1) % self.update_freq != 0:
#             return None

#         feasible = np.all((X >= problem_lb) & (X <= problem_ub), axis=1)

#         if np.count_nonzero(feasible) < 2:
#             return None

#         f_X_feasible = f_X[feasible]
#         X_feasible = X[feasible]
#         CP_feasible = CP[feasible]

#         elite_CPs = self.get_elite_CPs(f_X_feasible, X_feasible, CP_feasible)
#         return elite_CPs

#     def get_elite_CPs(self, f_X_feasible, X_feasible, CP_feasible):
#         return CP_feasible[
#             np.argsort(f_X_feasible)[:min(len(X_feasible), self.n_elite)]
#         ]

#     def update(self, elite_CPs):
#         scaled_CP = (elite_CPs - self.CP_lb) / (self.CP_ub - self.CP_lb)
#         eps = 1e-6
#         scaled_CP = np.clip(scaled_CP, eps, 1 - eps)

#         # Step 1 (Fit marginals):
#         eps = 1e-8          # numerical stability

#         weights = self.compute_weights(scaled_CP)

#         mu = np.average(scaled_CP, axis=0, weights=weights)

#         if weights is None:
#             var = np.var(scaled_CP, axis=0, ddof=1)
#         else:
#             diff = scaled_CP - mu
#             var = np.sum(weights[:, None] * diff**2, axis=0)
#             var = np.maximum(var, eps)
#         var = np.maximum(var, eps)

#         kappa = mu * (1 - mu) / var - 1
#         kappa = np.clip(kappa, eps, self.kappa_max)

#         a_new = mu * kappa
#         b_new = (1 - mu) * kappa

#         self.a = (1 - self.eta) * self.a + self.eta * a_new
#         self.b = (1 - self.eta) * self.b + self.eta * b_new

#     def compute_weights(self, scaled_CP):
#         return None

# class BetaMarginals_rank_weights_EDA(BetaMarginalsEDA):
#     def get_elite_CPs(self, f_X_feasible, X_feasible, CP_feasible):
#             # Sort all feasible particles by fitness (ascending for minimization)
#             CP_sorted = CP_feasible[np.argsort(f_X_feasible)]
#             return CP_sorted

#     def compute_weights(self, scaled_CP):
#             n = len(scaled_CP)

#             weights = np.arange(n, 0, -1, dtype=float)

#             weights /= weights.sum()

#             return weights

# class BetaMarginals_log_rank_weights_EDA(BetaMarginals_rank_weights_EDA):
#     def compute_weights(self, scaled_CP):
#             n = len(scaled_CP)

#             # ranks = 1, 2, ..., n
#             ranks = np.arange(1, n + 1)

#             weights = np.log(n + 1) - np.log(ranks)

#             # Numerical safety (should already be positive)
#             weights = np.maximum(weights, 0.0)

#             # Normalize
#             weights /= weights.sum()
#             return weights


# class BetaMarginalsGaussianCopulaEDA(BaseAdapter):
#     def __init__(self, hyper_params, search_space):
#         super().__init__(hyper_params, search_space)

#         self.initialize()

#     def initialize(self):
#         self.a = np.ones(shape=self.n_x, dtype=np.float32)
#         self.b = np.ones(shape=self.n_x, dtype=np.float32)
#         self.R = np.eye(N=self.n_x)

#         if "eta" in self.hyper_params.keys():
#             self.eta = self.hyper_params["eta"]

#         if "kappa_max" in self.hyper_params.keys():
#             self.kappa_max = self.hyper_params["kappa_max"]

#     def sample(self, N, **kwargs):
#         chol_eps = 1e-10 # used to ensure matrix is positive definite
#         d = len(self.a)
#         L = np.linalg.cholesky(self.R + chol_eps * np.eye(d))
#         Z = np.random.randn(N, len(self.a)) @ L.T
#         U = norm.cdf(Z)

#         samples = np.zeros((N, self.n_x))
#         for j in range(U.shape[1]):
#             samples[:, j] = beta.ppf(
#                 U[:, j],
#                 self.a[j],
#                 self.b[j]
#             )

#         samples = self.CP_lb + samples * (self.CP_ub - self.CP_lb)
#         return samples

#     def update(self, **kwargs):
#         X = kwargs["X"]
#         f_X = kwargs["f_X"]
#         CP = kwargs["CP"]

#         feasible = np.all(
#             (X >= kwargs["problem_lb"]) & (X <= kwargs["problem_ub"]), axis=1
#         )
#         f_X_feasible = f_X[feasible]
#         X_feasible = X[feasible]
#         CP_feasible = CP[feasible]

#         if len(X_feasible) < 2:
#             return False

#         samples = CP_feasible[np.argsort(f_X_feasible)[:min(len(X_feasible), 15)]]
#         # self.adapter.update(best_cp)
#         # self.CP = self.adapter.sample(N=self.n_s)


#         scaled_CP = (samples - self.CP_lb) / (self.CP_ub - self.CP_lb)
#         eps = 1e-6
#         scaled_CP = np.clip(scaled_CP, eps, 1 - eps)

#         # Step 1 (Fit marginals):
#         eps = 1e-8          # numerical stability

#         mu = np.mean(scaled_CP, axis=0)

#         var = np.var(scaled_CP, axis=0, ddof=1)
#         var = np.maximum(var, eps)

#         kappa = mu * (1 - mu) / var - 1
#         kappa = np.clip(kappa, eps, self.kappa_max)

#         a_new = mu * kappa
#         b_new = (1 - mu) * kappa

#         self.a = (1 - self.eta) * self.a + self.eta * a_new
#         self.b = (1 - self.eta) * self.b + self.eta * b_new

#         # Step 2 (Estimate Gaussian Copula):

#         # Transform into Gaussian space
#         Z = np.zeros_like(scaled_CP)

#         for j in range(scaled_CP.shape[1]):

#             u = beta.cdf(scaled_CP[:, j], self.a[j], self.b[j])

#             # Numerical safety
#             u = np.clip(u, 1e-12, 1 - 1e-12)

#             Z[:, j] = norm.ppf(u)  # Z is approximately multivariate normal

#         # Estimate Gaussian copula
#         R_new = np.corrcoef(Z.T)
#         self.R = (1 - self.eta) * self.R + self.eta * R_new
