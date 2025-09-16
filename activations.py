from abc import ABC, abstractmethod
from typing import Optional
import numpy as np

class Activation(ABC):
    def __init__(self):
        return

    @abstractmethod
    def func(self, *args): pass

    @abstractmethod
    def deriv(self, *args): pass


class Linear(Activation):
    def __init__(self):
        super().__init__()

    def func(self, z): return z

    def deriv(self, z): return np.ones_like(z)


class ReLU(Activation):
    def __init__(self):
        super().__init__()

    def func(self, z): return np.maximum(0, z)

    def deriv(self, z): return (z > 0).astype(float)


class Sigmoid(Activation):
    def __init__(self, max_exp_arg = 707):
        super().__init__()
        self.max_exp_arg = max_exp_arg #~709.78 for float64

    def func(self, z):
        clipped_z = np.clip(-z, -self.max_exp_arg, self.max_exp_arg)
        return 1 / (1 + np.exp(clipped_z))

    def deriv(self, z):
        s = self.func(z)
        return s * (1 - s)


class Softmax(Activation):
    def __init__(self):
        super().__init__()

    def func(self, z):
        # Per prevenire l'overflow numerico, si sottrae il massimo per stabilità
        e_z = np.exp(z - np.max(z, axis=-1, keepdims=True))
        return e_z / np.sum(e_z, axis=-1, keepdims=True)
    
    def deriv(self, z): return np.ones_like(z)