import numpy as np
from abc import ABC, abstractmethod


class BaseAdapter(ABC):
    def __init__(self, CP_space, **kwargs):
        self.CP_space = CP_space
        self.CP_lbs = CP_space.lower_bounds
        self.CP_ubs = CP_space.upper_bounds
        self.n_CP = CP_space.dimension

        self.initialize(**kwargs)

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
