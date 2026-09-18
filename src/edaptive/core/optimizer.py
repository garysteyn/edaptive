import numpy as np
from abc import ABC, abstractmethod

class BaseOptimizer(ABC):
    def __init__(self, **kwargs):
        self._timings = {}
        self.history = {
            "f_best": []
        }

        self.initialize(**kwargs)

        self.init_history()

    @property
    def CP(self):
        return self._CP

    @CP.setter
    def CP(self, value):
        self._CP = value

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
        return {
            name: getattr(self, name)
            for name in names
        }