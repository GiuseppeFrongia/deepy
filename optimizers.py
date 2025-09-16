from abc import ABC, abstractmethod
import numpy as np

class Optimizer(ABC):
    def __init__(self, learning_rate: float):
        self.learning_rate = learning_rate

    @abstractmethod
    def _update(self, params_and_grads: list[tuple[np.ndarray, np.ndarray]]): pass


class SGD(Optimizer):
    def __init__(self, learning_rate: float = 0.01):
        super().__init__(learning_rate)
        print(f"SGD with learning rate: {self.learning_rate}")

    def _update(self, params_and_grads: list[tuple[np.ndarray, np.ndarray]]):
        for param, grad in params_and_grads:
            param -= self.learning_rate * grad