from edaptive.problems.bbob import BBOBProblem
import math
import numpy as np
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
from edaptive.const import max_float
from edaptive.problems.benchmark_problem import BenchmarkProblem

def plot_benchmark_3d(
    benchmark_cls,
    resolution=200,
    elev=35,
    azim=-60,
    figsize=(10, 8),
    cmap="viridis",
    zlim=None,          # (zmin, zmax)
):
    """
    Plot a 2-dimensional benchmark function as a 3D surface.

    Parameters
    ----------
    benchmark_cls : class
        Benchmark class (not an instance). It must:
            - accept dimensions=2 in its constructor
            - have attributes lower and upper
            - be callable
    resolution : int, default=200
        Number of grid points per dimension.
    elev : float, default=35
        Elevation viewing angle.
    azim : float, default=-60
        Azimuth viewing angle.
    figsize : tuple, default=(10,8)
        Figure size.
    cmap : str, default="viridis"
        Matplotlib colormap.
    """

    # Instantiate a 2D benchmark
    f = benchmark_cls(dimensions=2)

    x = np.linspace(f.lower[0], f.upper[0], resolution)
    y = np.linspace(f.lower[1], f.upper[1], resolution)

    X, Y = np.meshgrid(x, y)

    Z = np.empty_like(X)

    for i in range(resolution):
        for j in range(resolution):
            Z[i, j] = f(np.array([X[i, j], Y[i, j]]))

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")

    surf = ax.plot_surface(
        X,
        Y,
        Z,
        cmap=cmap,
        linewidth=0,
        antialiased=True,
    )

    if zlim is not None:
        ax.set_zlim(*zlim)

    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.set_zlabel("$f(x)$")
    ax.set_title(benchmark_cls.__name__)

    ax.view_init(elev=elev, azim=azim)

    fig.colorbar(surf, shrink=0.6, aspect=15)

    plt.tight_layout()
    plt.show()


class Schwefel1(BenchmarkProblem):
    """
    Schwefel 1 benchmark function.

    Parameters
    ----------
    dimensions : int
        Number of decision variables.
    alpha : float, optional
        Exponent applied to the sum of squares.
        Default is sqrt(pi).

    Domain
    ------
    x_i ∈ [-100, 100]

    Global Optimum
    --------------
    f(0, ..., 0) = 0
    """

    def __init__(self, dimensions, alpha=np.sqrt(np.pi)):
        super().__init__(
            dimensions=dimensions,
        )
        self.alpha = alpha

        self.lower = np.full(dimensions, -100.0)
        self.upper = np.full(dimensions, 100.0)

        self.global_optimum = np.zeros(dimensions)
        self.global_optimum_value = 0.0

    def _evaluate(self, x):
        return np.sum(x ** 2) ** self.alpha

class Ripple25(BenchmarkProblem):
    """
    Ripple 25 benchmark function.

    Domain
    ------
    x_i ∈ [0, 1]

    Global Optimum
    --------------
    x_i = 0.1 for all i

    f(x*) = -dimensions
    """

    def __init__(self, dimensions):
        super().__init__(
            dimensions=dimensions,
        )

        self.lower = np.zeros(dimensions)
        self.upper = np.ones(dimensions)

        self.global_optimum = np.full(dimensions, 0.1)
        self.global_optimum_value = -float(dimensions)

    def _evaluate(self, x):
        u = -2.0 * np.log(2.0) * ((x - 0.1) / 0.8) ** 2
        v = np.sin(5.0 * np.pi * x) ** 6

        return np.sum(-np.exp(u) * v)

class Exponential(BenchmarkProblem):
    """
    Exponential benchmark function.

    f(x) = -exp(-0.5 * sum(x_i^2))

    Domain
    ------
    x_i ∈ [-1, 1]

    Global Optimum
    --------------
    x* = (0, ..., 0)

    f(x*) = -1
    """

    def __init__(self, dimensions):
        super().__init__(
            dimensions=dimensions,
        )

        self.lower = np.full(dimensions, -1.0)
        self.upper = np.full(dimensions, 1.0)

        self.global_optimum = np.zeros(dimensions)
        self.global_optimum_value = -1.0

    def _evaluate(self, x):
        return -np.exp(-0.5 * np.sum(x**2))

class NeedleEye(BenchmarkProblem):
    """
    Needle Eye benchmark function.

    Domain
    ------
    x_i ∈ [-10, 10]

    Global Optimum
    --------------
    If all |x_i| < 1e-4,

        f(x) = 1

    Otherwise,

        if all |x_i| > 1e-4:
            f(x) = sum(100 + |x_i|)

        else:
            f(x) = 0
    """

    def __init__(self, dimensions):
        super().__init__(
            dimensions=dimensions,
        )

        self.lower = np.full(dimensions, -10.0)
        self.upper = np.full(dimensions, 10.0)

        self.global_optimum = np.zeros(dimensions)
        self.global_optimum_value = 1.0

    def _evaluate(self, x):
        eye = 1e-4
        a = np.abs(x)

        # All variables are inside the eye
        if np.all(a < eye):
            return 1.0

        # All variables are outside the eye
        elif np.all(a > eye):
            return np.sum(100.0 + a)

        # Some inside and some outside
        else:
            return 0.0

class Step3(BenchmarkProblem):
    """
    Step Function No. 3 benchmark function.

    f(x) = sum(floor(x_i^2))

    Domain
    ------
    x_i ∈ [-100, 100]

    Global Optimum
    --------------
    x* = (0, ..., 0)

    f(x*) = 0
    """

    def __init__(self, dimensions):
        super().__init__(
            dimensions=dimensions,
        )

        self.lower = np.full(dimensions, -100.0)
        self.upper = np.full(dimensions, 100.0)

        self.global_optimum = np.zeros(dimensions)
        self.global_optimum_value = 0.0

    def _evaluate(self, x):
        return np.sum(np.floor(x**2))


class GeneralizedGiunta(BenchmarkProblem):
    """
    Generalized Giunta benchmark function.

    Domain
    ------
    x_i ∈ [-1, 1]

    Notes
    -----
    Generalization of the 2-dimensional Giunta function.

    Reference
    ---------
    https://arxiv.org/pdf/1308.4008.pdf (Equation 57)
    """

    def __init__(self, dimensions):
        super().__init__(
            dimensions=dimensions,
        )

        self.lower = np.full(dimensions, -1.0)
        self.upper = np.full(dimensions, 1.0)

        # Global optimum not specified in the original R implementation.
        self.global_optimum = None
        self.global_optimum_value = None

    def _evaluate(self, x):
        a = 1.067 * x - 1.0
        b = np.sin(a)

        return 0.6 + np.sum(
            b +
            b**2 +
            0.02 * np.sin(4.0 * a)
        )

class GeneralizedPaviani(BenchmarkProblem):
    """
    Generalized Paviani benchmark function.

    f(x) = sum(log(10 - x_i)^2 + log(x_i - 2)^2)
           - (prod(x_i))^0.2

    Domain
    ------
    x_i ∈ [2.001, 9.999]

    Notes
    -----
    This is the generalized n-dimensional version of the Paviani function.
    The original R implementation does not specify the global optimum.
    """

    def __init__(self, dimensions):
        super().__init__(
            dimensions=dimensions,
        )

        self.lower = np.full(dimensions, 2.001)
        self.upper = np.full(dimensions, 9.999)

        # Not specified in the original implementation
        self.global_optimum = None
        self.global_optimum_value = None

    def _evaluate(self, x):
        if np.any((x <= 2) | (x >= 10)):
            return np.inf
        return (
            np.sum(
                np.log(10.0 - x) ** 2 +
                np.log(x - 2.0) ** 2
            )
            - np.prod(x) ** 0.2
        )

class Brown(BenchmarkProblem):
    """
    Brown benchmark function.

    f(x) = sum(
        (x_i^2)^(x_{i+1}^2 + 1)
        + (x_{i+1}^2)^(x_i^2 + 1)
    )

    Domain
    ------
    x_i ∈ [-1, 4]

    Global Optimum
    --------------
    x* = (0, ..., 0)

    f(x*) = 0
    """

    def __init__(self, dimensions):
        super().__init__(
            dimensions=dimensions,
        )

        self.lower = np.full(dimensions, -1.0)
        self.upper = np.full(dimensions, 4.0)

        self.global_optimum = np.zeros(dimensions)
        self.global_optimum_value = 0.0

    def _evaluate(self, x):
        x1 = x[:-1]
        x2 = x[1:]

        return np.sum(
            (x1**2) ** (x2**2 + 1.0)
            + (x2**2) ** (x1**2 + 1.0)
        )

class CosineMixture_OG(BenchmarkProblem):

    def __init__(self, dimensions):
        super().__init__(
            dimensions=dimensions,
        )

        self.lower = np.full(dimensions, -1.0)
        self.upper = np.full(dimensions, 1.0)

        self.global_optimum = np.zeros(dimensions)
        self.global_optimum_value = -0.1 * dimensions

    def _evaluate(self, x):
        return (
            -0.1 * np.sum(np.cos(5.0 * np.pi * x))
            + np.sum(x**2)
        )

class CosineMixture(BenchmarkProblem):

    def __init__(self, dimensions):
        super().__init__(
            dimensions=dimensions,
        )

        self.lower = np.full(dimensions, -1.0)
        self.upper = np.full(dimensions, 1.0)

        self.global_optimum = np.zeros(dimensions)
        self.global_optimum_value = -0.1 * dimensions

    def _evaluate(self, x):
        return (
            -0.1 * np.sum(np.cos(5.0 * np.pi * x))
            - np.sum(x**2)
        )

class Mishra07(BenchmarkProblem):
    """
    Mishra No. 7 (Factorial) benchmark function.

    f(x) = (prod(x) - D!)^2

    where D is the number of dimensions.

    Domain
    ------
    x_i ∈ [-10, 10]

    Global Optimum
    --------------
    Any point satisfying

        prod(x) = D!

    is a global minimizer, with

        f(x*) = 0.
    """

    def __init__(self, dimensions):
        super().__init__(
            dimensions=dimensions,
        )

        self.lower = np.full(dimensions, -10.0)
        self.upper = np.full(dimensions, 10.0)

        # There are infinitely many global optima.
        self.global_optimum = None
        self.global_optimum_value = 0.0

    def _evaluate(self, x):
        return (np.prod(x) - math.factorial(self.dimensions)) ** 2

class Mishra01(BenchmarkProblem):
    """
    Mishra No. 1 benchmark function.

    Reference
    ---------
    http://infinity77.net/global_optimization/test_functions_nd_M.html#go_benchmark.Mishra01

    Domain
    ------
    x_i ∈ [0, 1]

    Notes
    -----
    This implementation follows the original R code exactly.
    The last decision variable is ignored.
    """

    def __init__(self, dimensions):
        super().__init__(
            dimensions=dimensions,
        )

        self.lower = np.zeros(dimensions)
        self.upper = np.ones(dimensions)

        # Not specified in the R implementation
        self.global_optimum = None
        self.global_optimum_value = None

    def _evaluate(self, x):
        n = self.dimensions

        # Equivalent to x[-n] in R (all elements except the last)
        xn = n - np.sum(x[:-1])

        return (1.0 + xn) ** xn

class GeneralizedPrice2(BenchmarkProblem):
    """
    Generalized Price No. 2 benchmark function.

    f(x) = 1 + sum(sin(x_i)^2) - 0.1 * exp(-sum(x_i^2))

    Domain
    ------
    x_i ∈ [-10, 10]

    Global Optimum
    --------------
    x* = (0, ..., 0)

    f(x*) = 0.9
    """

    def __init__(self, dimensions):
        super().__init__(
            dimensions=dimensions,
        )

        self.lower = np.full(dimensions, -10.0)
        self.upper = np.full(dimensions, 10.0)

        self.global_optimum = np.zeros(dimensions)
        self.global_optimum_value = 0.9

    def _evaluate(self, x):
        return (
            1.0
            + np.sum(np.sin(x) ** 2)
            - 0.1 * np.exp(-np.sum(x ** 2))
        )

class GeneralizedEggCrate(BenchmarkProblem):
    """
    Generalized Egg Crate benchmark function.

    f(x) = sum(x_i^2) + 24 * sum(sin(x_i)^2)

    Domain
    ------
    x_i ∈ [-5, 5]

    Global Optimum
    --------------
    x* = (0, ..., 0)

    f(x*) = 0
    """

    def __init__(self, dimensions):
        super().__init__(
            dimensions=dimensions,
        )

        self.lower = np.full(dimensions, -5.0)
        self.upper = np.full(dimensions, 5.0)

        self.global_optimum = np.zeros(dimensions)
        self.global_optimum_value = 0.0

    def _evaluate(self, x):
        return (
            np.sum(x ** 2)
            + 24.0 * np.sum(np.sin(x) ** 2)
        )

class Rosenbrock(BenchmarkProblem):
    """
    Rosenbrock benchmark function.

    f(x) = sum(
        100 * (x_{i+1} - x_i^2)^2
        + (x_i - 1)^2
    )

    Domain
    ------
    x_i ∈ [-30, 30]

    Global Optimum
    --------------
    x* = (1, ..., 1)

    f(x*) = 0
    """

    def __init__(self, dimensions):
        super().__init__(
            dimensions=dimensions,
        )

        self.lower = np.full(dimensions, -30)
        self.upper = np.full(dimensions, 30)

        self.global_optimum = np.ones(dimensions)
        self.global_optimum_value = 0.0

    def _evaluate(self, x):
        return np.sum(
            100.0 * (x[1:] - x[:-1] ** 2) ** 2
            + (x[:-1] - 1.0) ** 2
        )

# Version used in the ELA uses log, not log_10
class Pinter2(BenchmarkProblem):
    """
    Pinter 2 benchmark function.

    f(x) = a + b + c

    where

    a = sum(i * x_i^2)

    b = sum(
        20 * i *
        sin(
            x_{i-1} * sin(x_i)
            + sin(x_{i+1})
        )^2
    )

    c = sum(
        i * log(
            1 + i *
            (x_{i-1}^2 - 2*x_i + 3*x_{i+1}
             - cos(x_i) + 1)^2
        )
    )

    The function uses cyclic boundary conditions.

    Domain
    ------
    x_i ∈ [-10, 10]

    Global Optimum
    --------------
    x* = (0, ..., 0)

    f(x*) = 0
    """

    def __init__(self, dimensions):
        super().__init__(
            dimensions=dimensions,
        )

        self.lower = np.full(dimensions, -10.0)
        self.upper = np.full(dimensions, 10.0)

        self.global_optimum = np.zeros(dimensions)
        self.global_optimum_value = 0.0

    def _evaluate(self, x):
        n = len(x)

        # R: a = sum(1:n * x^2)
        i = np.arange(1, n + 1, dtype=float)
        a = np.sum(i * x ** 2)

        # Cyclically extended vector:
        # R: z = c(x[n], x, x[1])
        z = np.concatenate(([x[-1]], x, [x[0]]))

        # R: i = 2:(length(z) - 1)
        #
        # In zero-based Python indexing, these correspond to
        # z[1:-1], with the mathematical indices 1,...,n.
        i = np.arange(1, n + 1, dtype=float)

        b = np.sum(
            20.0
            * i
            * np.sin(
                z[:-2] * np.sin(z[1:-1])
                + np.sin(z[2:])
            ) ** 2
        )

        c = np.sum(
            i
            * np.log(
                1.0
                + i
                * (
                    z[:-2] ** 2
                    - 2.0 * z[1:-1]
                    + 3.0 * z[2:]
                    - np.cos(z[1:-1])
                    + 1.0
                ) ** 2
            )
        )

        return a + b + c

class Pinter2_OG(BenchmarkProblem):
    """
    Pinter 2 benchmark function.

    f(x) = a + b + c

    where

    a = sum(i * x_i^2)

    b = sum(
        20 * i *
        sin(
            x_{i-1} * sin(x_i)
            + sin(x_{i+1})
        )^2
    )

    c = sum(
        i * log(
            1 + i *
            (x_{i-1}^2 - 2*x_i + 3*x_{i+1}
             - cos(x_i) + 1)^2
        )
    )

    The function uses cyclic boundary conditions.

    Domain
    ------
    x_i ∈ [-10, 10]

    Global Optimum
    --------------
    x* = (0, ..., 0)

    f(x*) = 0
    """

    def __init__(self, dimensions):
        super().__init__(
            dimensions=dimensions,
        )

        self.lower = np.full(dimensions, -10.0)
        self.upper = np.full(dimensions, 10.0)

        self.global_optimum = np.zeros(dimensions)
        self.global_optimum_value = 0.0

    def _evaluate(self, x):
        n = len(x)

        # R: a = sum(1:n * x^2)
        i = np.arange(1, n + 1, dtype=float)
        a = np.sum(i * x ** 2)

        # Cyclically extended vector:
        # R: z = c(x[n], x, x[1])
        z = np.concatenate(([x[-1]], x, [x[0]]))

        # R: i = 2:(length(z) - 1)
        #
        # In zero-based Python indexing, these correspond to
        # z[1:-1], with the mathematical indices 1,...,n.
        i = np.arange(1, n + 1, dtype=float)

        b = np.sum(
            20.0
            * i
            * np.sin(
                z[:-2] * np.sin(z[1:-1])
                + np.sin(z[2:])
            ) ** 2
        )

        c = np.sum(
            i
            * np.log10(
                1.0
                + i
                * (
                    z[:-2] ** 2
                    - 2.0 * z[1:-1]
                    + 3.0 * z[2:]
                    - np.cos(z[1:-1])
                    + 1.0
                ) ** 2
            )
        )

        return a + b + c

class Qing(BenchmarkProblem):
    """
    Qing benchmark function.

    f(x) = sum((x_i^2 - i)^2)

    Domain
    ------
    x_i ∈ [-500, 500]

    Global Optimum
    --------------
    x_i = ±sqrt(i),  i = 1, ..., n

    f(x*) = 0
    """

    def __init__(self, dimensions):
        super().__init__(
            dimensions=dimensions,
        )

        self.lower = np.full(dimensions, -500.0)
        self.upper = np.full(dimensions, 500.0)

        self.global_optimum = np.sqrt(
            np.arange(1, dimensions + 1)
        )
        self.global_optimum_value = 0.0

    def _evaluate(self, x):
        i = np.arange(1, self.dimensions + 1)

        return np.sum(
            (x ** 2 - i) ** 2
        )

class BBOB_FID2_IID1(BBOBProblem):
    """
    BBOB FID 2 (Ellipsoid Separable) IID 1
    """

    def __init__(self, dimensions):
        super().__init__(
            dimensions=dimensions,
            function_id=2,
            instance_id=1,
        )

class BBOB_FID6_IID1(BBOBProblem):
    """
    BBOB FID 6 (Attractive Sector) IID 1
    """

    def __init__(self, dimensions):
        super().__init__(
            dimensions=dimensions,
            function_id=6,
            instance_id=1,
        )

class BBOB_FID16_IID1(BBOBProblem):
    """
    BBOB FID 16 (Weierstrass) IID 1
    """

    def __init__(self, dimensions):
        super().__init__(
            dimensions=dimensions,
            function_id=16,
            instance_id=1,
        )

class BBOB_FID17_IID2(BBOBProblem):
    """
    BBOB FID 17 (Schaffer F7, Condition 10) IID 2
    """

    def __init__(self, dimensions):
        super().__init__(
            dimensions=dimensions,
            function_id=17,
            instance_id=2,
        )

class DropWave(BenchmarkProblem):
    """
    Drop-Wave benchmark function.

    f(x) = -(1 + cos(12 * sqrt(sum(x_i^2))))
           / (2 + 0.5 * sum(x_i^2))
    """

    def __init__(self, dimensions):
        super().__init__(
            dimensions=dimensions,
        )

        self.lower = np.full(dimensions, -5.12)
        self.upper = np.full(dimensions, 5.12)

        self.global_optimum = np.zeros(dimensions)
        self.global_optimum_value = -1.0

    def _evaluate(self, x):
        sumsqr = np.sum(x ** 2)

        return -(
            1.0 + np.cos(12.0 * np.sqrt(sumsqr))
        ) / (
            2.0 + 0.5 * sumsqr
        )

class BonyadiMichalewicz(BenchmarkProblem):
    """
    Bonyadi-Michalewicz benchmark function.

    f(x) = prod(x_i + 1) / prod((x_i - 1)^2 + 1)

    Domain
    ------
    x_i ∈ [-5, 5]
    """

    def __init__(self, dimensions):
        super().__init__(
            dimensions=dimensions,
        )

        self.lower = np.full(dimensions, -5.0)
        self.upper = np.full(dimensions, 5.0)

    def _evaluate(self, x):
        a = np.prod(x + 1.0)
        b = np.prod((x - 1.0) ** 2 + 1.0)

        return a / b

class Discus(BenchmarkProblem):
    """
    Discus benchmark function.

    f(x) = 10^6 * x_1^2 + sum_{i=2}^n x_i^2

    Domain
    ------
    x_i ∈ [-100, 100]
    """

    def __init__(self, dimensions):
        super().__init__(
            dimensions=dimensions,
        )

        self.lower = np.full(dimensions, -100.0)
        self.upper = np.full(dimensions, 100.0)

        self.global_optimum = np.zeros(dimensions)
        self.global_optimum_value = 0.0

    def _evaluate(self, x):
        return (
            1.0e6 * x[0] ** 2
            + np.sum(x[1:] ** 2)
        )

class Elliptic(BenchmarkProblem):
    """
    Elliptic benchmark function.

    f(x) = sum_{i=1}^n 10^(6(i-1)/(n-1)) * x_i^2

    Domain
    ------
    x_i ∈ [-100, 100]
    """

    def __init__(self, dimensions):
        super().__init__(
            dimensions=dimensions,
        )

        self.lower = np.full(dimensions, -100.0)
        self.upper = np.full(dimensions, 100.0)

        self.global_optimum = np.zeros(dimensions)
        self.global_optimum_value = 0.0

    def _evaluate(self, x):
        i = np.arange(self.dimensions)

        return np.sum(
            10.0 ** (
                6.0 * i / (self.dimensions - 1.0)
            )
            * x ** 2
        )

def main():
    plot_benchmark_3d(NeedleEye)

if __name__ == "__main__":
    main()