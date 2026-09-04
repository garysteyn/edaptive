import numpy as np
from edaptive.core.adapter import EDAAdapter
from scipy.stats import beta, norm

class BetaMarginalsEDA(EDAAdapter):
    def __init__(self, search_space, eta, kappa_max, n_elite, update_freq, rng):
        super().__init__(search_space)
        self.initialize(eta, kappa_max, n_elite, update_freq, rng)

    def required_data(self):
        return ["problem_lb", "problem_ub", "t", "X", "f_X", "CP"]

    def initialize(self, eta, kappa_max, n_elite, update_freq, rng):
        self.rng = rng

        self.a = np.ones(shape=self.n_x, dtype=np.float32) 
        self.b = np.ones(shape=self.n_x, dtype=np.float32)

        self.eta = eta
        self.kappa_max = kappa_max
        self.n_elite = int(n_elite)
        self.update_freq = update_freq

    def sample(self, N):
        samples = np.column_stack(
            [
                beta.rvs(
                    self.a[j],
                    self.b[j],
                    size=N,
                    random_state=self.rng
                )
                for j in range(self.n_x)
            ]
        )
        samples = self.CP_lb + samples * (self.CP_ub - self.CP_lb)
        return samples

    def check_update(self, problem_lb, problem_ub, t, X, f_X, CP):
        if (t + 1) % self.update_freq != 0:
            return None

        feasible = np.all((X >= problem_lb) & (X <= problem_ub), axis=1)
        if feasible.sum() < 2:
            return None

        f_X_feasible = f_X[feasible]
        X_feasible = X[feasible]
        CP_feasible = CP[feasible]

        elite_CPs = self.get_elite_CPs(f_X_feasible, X_feasible, CP_feasible)
        return elite_CPs

    def get_elite_CPs(self, f_X_feasible, X_feasible, CP_feasible):
        return CP_feasible[np.argsort(f_X_feasible)[:min(len(X_feasible), self.n_elite)]]

    def update(self, elite_CPs):
        scaled_CP = (elite_CPs - self.CP_lb) / (self.CP_ub - self.CP_lb)
        eps = 1e-6
        scaled_CP = np.clip(scaled_CP, eps, 1 - eps)

        # Step 1 (Fit marginals):
        eps = 1e-8          # numerical stability

        weights = self.compute_weights(scaled_CP)

        mu = np.average(scaled_CP, axis=0, weights=weights)

        if weights is None: 
            var = np.var(scaled_CP, axis=0, ddof=1)
        else:
            diff = scaled_CP - mu
            var = np.sum(weights[:, None] * diff**2, axis=0)
            var = np.maximum(var, eps)
        var = np.maximum(var, eps)

        kappa = mu * (1 - mu) / var - 1
        kappa = np.clip(kappa, eps, self.kappa_max)

        a_new = mu * kappa
        b_new = (1 - mu) * kappa

        self.a = (1 - self.eta) * self.a + self.eta * a_new
        self.b = (1 - self.eta) * self.b + self.eta * b_new

    def compute_weights(self, scaled_CP):
        return None

class BetaMarginals_rank_weights_EDA(BetaMarginalsEDA):
    def get_elite_CPs(self, f_X_feasible, X_feasible, CP_feasible):
            # Sort all feasible particles by fitness (ascending for minimization)
            CP_sorted = CP_feasible[np.argsort(f_X_feasible)]
            return CP_sorted

    def compute_weights(self, scaled_CP):
            n = len(scaled_CP)

            weights = np.arange(n, 0, -1, dtype=float)

            weights /= weights.sum()

            return weights

class BetaMarginals_log_rank_weights_EDA(BetaMarginals_rank_weights_EDA):
    def compute_weights(self, scaled_CP):
            n = len(scaled_CP)

            # ranks = 1, 2, ..., n
            ranks = np.arange(1, n + 1)

            weights = np.log(n + 1) - np.log(ranks)

            # Numerical safety (should already be positive)
            weights = np.maximum(weights, 0.0)

            # Normalize
            weights /= weights.sum()
            return weights


    
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

#         feasible = np.all((X >= kwargs["problem_lb"]) & (X <= kwargs["problem_ub"]), axis=1)
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