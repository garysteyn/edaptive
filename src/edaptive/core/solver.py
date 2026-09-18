import numpy as np
from abc import ABC, abstractmethod

class BaseSolver(ABC):
    def __init__(self, optimizer_type, adapter_type=None, **kwargs):
        self.optimizer_type = optimizer_type
        self.adapter_type = adapter_type
        
        self.initialize(**kwargs)

    @abstractmethod
    def initialize(self, **kwargs):
        pass

    @abstractmethod
    def run(self, max_iterations=5000):
        pass