import numpy as np
from abc import ABC, abstractmethod


class BaseAdapter(ABC):
    def __init__(self, param_space, **kwargs):
        self.param_space = param_space
        self.CP_lb = param_space.lower_bounds
        self.CP_ub = param_space.upper_bounds
        self.n_params = param_space.dimension

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
