import numpy as np
from abc import ABC, abstractmethod


class BaseAdapter(ABC):
    def __init__(self, CP_space, **kwargs):
        self.CP_space = CP_space
        self.CP_lbs = CP_space.lower_bounds
        self.CP_ubs = CP_space.upper_bounds
        self.n_CPs = CP_space.dimension
        self.CP_names = tuple(parameter.name for parameter in CP_space)

        self.initialize(**kwargs)

    def select_own_CPs(self, adaptable_CPs, adaptable_CP_names):
        """Columns of an optimizer's ``adaptable_CPs`` for this adapter's CPs, in
        ``CP_space`` order."""
        columns = [adaptable_CP_names.index(name) for name in self.CP_names]
        return adaptable_CPs[:, columns]

    # Data required of the optimizer (to perform an update)
    def required_data(self):
        pass

    @abstractmethod
    def initialize(self, **kwargs):
        pass

    def update(self):
        pass


class EDAAdapter(BaseAdapter):
    @abstractmethod
    def sample(self, N):
        pass
