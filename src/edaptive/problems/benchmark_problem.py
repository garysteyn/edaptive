import numpy as np
from abc import ABC, abstractmethod
from edaptive.const import max_float

class BenchmarkProblem(ABC):
    def __init__(self, dimensions):
        if dimensions <= 0:
            raise ValueError("dimensions must be a positive integer")
        self.dimensions = dimensions

    def __call__(self, x):
        x = np.asarray(x, dtype=float)

        if x.shape != (self.dimensions,):
            raise ValueError(
                f"x must have shape ({self.dimensions},)"
            )

        with np.errstate(
            over="ignore",
            invalid="ignore",
            divide="ignore"
        ):
            result = self._evaluate(x)
        # result = self._evaluate(x)
        return np.nan_to_num(
            result,
            nan=max_float,
            posinf=max_float,
            neginf=-max_float
        )
    
    @abstractmethod
    def _evaluate(self, x):
        """Evaluate the benchmark function."""
        pass
