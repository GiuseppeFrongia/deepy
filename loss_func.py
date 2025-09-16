from abc import ABC, abstractmethod
from typing import Optional
import numpy as np

class LossFunc(ABC):
    def __init__(self): pass

    @abstractmethod
    def _loss(self, y_hat: np.ndarray, targets: np.ndarray) -> float: pass

    @abstractmethod
    def _grad(self, y_hat: np.ndarray, targets: np.ndarray) -> np.ndarray: pass


class MeanSquaredError(LossFunc):
    def __init__(self): pass

    def _loss(self, y_hat: np.ndarray, targets: np.ndarray) -> float:
        if y_hat.shape != targets.shape:
            raise ValueError(f"Shape mismatch: predictions {y_hat.shape} vs targets {targets.shape}")
        
        return np.mean(np.square(y_hat - targets))
    
    def _grad(self, y_hat: np.ndarray, targets: np.ndarray) -> np.ndarray:
        return 2 * (y_hat - targets) / y_hat.shape[0]


class BinaryCrossEntropy(LossFunc):
    def __init__(self): pass

    def _loss(self, y_hat: np.ndarray, targets: np.ndarray) -> float:
        if y_hat.shape != targets.shape:
            raise ValueError(f"Shape mismatch: predictions {y_hat.shape} vs targets {targets.shape}")
        y_hat = np.clip(y_hat, 1e-15, 1 - 1e-15)
        
        return -np.mean(targets * np.log(y_hat) + (1 - targets) * np.log(1 - y_hat))

    def _grad(self, y_hat: np.ndarray, targets: np.ndarray) -> np.ndarray:
        grad = (y_hat - targets) / (y_hat * (1 - y_hat))
        
        return grad / y_hat.shape[0]


class CategoricalCrossEntropy(LossFunc):
    def __init__(self): pass

    def _loss(self, y_hat: np.ndarray, targets: np.ndarray) -> float:
        y_hat = np.clip(y_hat, 1e-15, 1 - 1e-15)
        n_samples = y_hat.shape[0]
        
        if targets.ndim == 2 and targets.shape == y_hat.shape:
            #one-hot encoding
            loss = -np.sum(targets * np.log(y_hat)) / n_samples
        elif targets.ndim == 1 or (targets.ndim == 2 and targets.shape[1] == 1):
            #sparse
            n_samples = y_hat.shape[0]
            if targets.ndim == 2 and targets.shape[1] == 1:
                targets = targets.flatten()
            
            loss = -np.sum(np.log(y_hat[np.arange(n_samples), targets])) / n_samples
        else:
            raise ValueError(f"Shape mismatch: predictions {y_hat.shape} vs targets {targets.shape}")
        
        return loss

    def _grad(self, y_hat: np.ndarray, targets: np.ndarray) -> np.ndarray:

        n_samples = y_hat.shape[0]
        grad = y_hat.copy()

        #one-hot encoding
        if targets.ndim == 2 and targets.shape == y_hat.shape:
            grad -= targets
        #sparse
        elif targets.ndim == 1 or (targets.ndim == 2 and targets.shape[1] == 1):
            if targets.ndim == 2 and targets.shape[1] == 1:
                targets = targets.flatten()
            
            grad[np.arange(n_samples), targets] -= 1
        else:
            raise ValueError("Target format not supported for CrossEntropy. Must be one-hot or labels.")
        
        return grad / n_samples