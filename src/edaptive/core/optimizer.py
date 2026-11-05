from abc import ABC, abstractmethod


class BaseOptimizer(ABC):
    """Base class for population-based optimizers.

    Adaptable CPs are the per-individual control parameters that may change during
    a run (PSO: w, c_1, c_2). They are stored in ``adaptable_CPs``, an array of
    shape ``(n_s, len(adaptable_CP_names))`` with one row per individual and one
    column per name in ``adaptable_CP_names``. Other CPs, such as the swarm size
    ``n_s``, are fixed when the optimizer is built. Which adaptable CPs a solver
    actually adapts is set by its config (``"fixed"`` vs ``"adapted"``).
    """

    adaptable_CP_names: tuple[str, ...] = ()

    def __init__(self, **kwargs):
        self._timings = {}
        self.history = {"f_best": []}

        self.initialize(**kwargs)

        self.init_history()

    @property
    def adaptable_CPs(self):
        return self._adaptable_CPs

    def column_indices(self, names):
        """Columns of ``adaptable_CPs`` that hold the CPs called ``names``."""
        unknown = [name for name in names if name not in self.adaptable_CP_names]
        if unknown:
            raise ValueError(
                f"{type(self).__name__} has no adaptable CPs {unknown}; "
                f"its adaptable CPs are {self.adaptable_CP_names}"
            )
        return [self.adaptable_CP_names.index(name) for name in names]

    def set_adaptable_CPs(self, names, values):
        """Overwrite the columns for the CPs called ``names`` with ``values``.

        ``values`` has shape ``(n_s, len(names))``. Columns not named keep their
        values.
        """
        adaptable_CPs = self._adaptable_CPs.copy()
        adaptable_CPs[:, self.column_indices(names)] = values
        self._adaptable_CPs = adaptable_CPs

    @abstractmethod
    def initialize(self, **kwargs):
        pass

    @abstractmethod
    def init_history(self):
        pass

    @abstractmethod
    def step(self):
        pass

    def get_required_data(self, names):
        return {name: getattr(self, name) for name in names}
