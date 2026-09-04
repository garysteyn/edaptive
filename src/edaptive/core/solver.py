import numpy as np
from abc import ABC, abstractmethod

class BaseSolver(ABC):
    def __init__(self, optimizer_type, adapter_type):
        self.optimizer_type = optimizer_type
        self.adapter_type = adapter_type

    @abstractmethod
    def initialize(self, problem, hyper_params):
        pass

    @abstractmethod
    def run(self, max_iterations=5000):
        pass