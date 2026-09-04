import numpy as np
from abc import ABC, abstractmethod

class BaseOptimizer(ABC):
    def __init__(self):
        self._timings = {}
        self.history = {
            "f_best": []
        }

    @property
    def CP(self):
        return self._CP

    @CP.setter
    def CP(self, value):
        self._CP = value

    @abstractmethod
    def get_required_data(self, names):
        return {
            name: getattr(self, name)
            for name in names
        }

    @abstractmethod
    def initialize(self):
        pass

    @abstractmethod
    def step(self):
        pass