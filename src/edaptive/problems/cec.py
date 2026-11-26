import importlib

import numpy as np

from edaptive.problems.benchmark_problem import BenchmarkProblem


class CECProblem(BenchmarkProblem):
    def __init__(self, dimensions, suite, function_id):
        super().__init__(
            dimensions=dimensions,
        )

        module = importlib.import_module(f"opfunu.cec_based.cec{suite}")

        function_class = getattr(
            module,
            f"F{function_id}{suite}",
        )

        self._function = function_class(
            ndim=dimensions,
        )

        self.lower = np.asarray(self._function.lb)
        self.upper = np.asarray(self._function.ub)

    def _evaluate(self, x):
        # opfunu evaluates one solution at a time.
        x = np.asarray(x)

        if x.ndim == 1:
            return self._function.evaluate(x)

        if x.ndim == 2:
            return np.asarray([self._function.evaluate(xi) for xi in x])

        raise ValueError(f"Expected a 1D or 2D array, got shape {x.shape}.")
