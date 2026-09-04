import numpy as np
from abc import ABC, abstractmethod

# TODO: change "search space" to "parameter space"
class BaseAdapter(ABC):
    def __init__(self, search_space):
        self.search_space = search_space
        self.CP_lb = search_space.lower_bounds
        self.CP_ub = search_space.upper_bounds
        self.n_x = search_space.dimension

    # Data required of the optimizer (to perform an update)
    @abstractmethod
    def required_data(self):
        pass

    @abstractmethod
    def initialize(self):
        pass

    @abstractmethod
    def sample(self, N):
        pass

    @abstractmethod
    def update(self):
        pass
