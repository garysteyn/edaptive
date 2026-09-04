import numpy as np
import cocoex
from edaptive.problems.benchmark_problem import BenchmarkProblem


class BBOBProblem(BenchmarkProblem):

    def __init__(self, dimensions, function_id, instance_id):
        super().__init__(
            dimensions=dimensions,
        )

        suite = cocoex.Suite("bbob", "", "")

        self._function = suite.get_problem_by_function_dimension_instance(
            function_id,
            dimensions,
            instance_id,
        )

        self.lower = np.asarray(self._function.lower_bounds)
        self.upper = np.asarray(self._function.upper_bounds)

    def _evaluate(self, x):
        return self._function(x)