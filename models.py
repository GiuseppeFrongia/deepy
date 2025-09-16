#from abc import ABC, abstractmethod
from typing import Optional, Union
import numpy as np
from tqdm import tqdm
from .layers.core import Layer, BatchNormalization, Sampling
from .optimizers import Optimizer

class Model():
    def __init__(self, layers: list[Layer] = None):
        self.layers: list[Layer] = []
        self.is_build: bool = False
        self.data: Optional[dict] = None
        self.optimizer: Optional[Optimizer] = None
        if layers:
            for layer in layers:
                self.layers.append(layer)

    def __add__(self, other: 'Model') -> 'Model':
        if not isinstance(other, Model):
            raise TypeError(f"Unsupported operand type(s) for +: 'Model' and '{type(other).__name__}'")
        
        comb_layers = self.layers + other.layers
        return Model(comb_layers)

    def _build(self, input_shape):
        layer_shape = input_shape
        for layer in self.layers:
            layer._build(layer_shape)
            layer_shape = layer.output_shape
        self.is_build = True
        print("Model has been build.")

    def _forward(self, X: np.ndarray, y: np.ndarray):
        x = X
        for layer in self.layers[:-1]: x = layer._forward(x)
        return self.layers[-1]._forward(x, y)

    def _backprop(self, y_hat, y_batch):
        grad = self.layers[-1]._backprop(y_hat, y_batch)
        for layer in reversed(self.layers[:-1]):
            grad = layer._backprop(grad)
        return grad
    
    def _batch_update(self, X_batch, y_batch):
        y_hat, loss = self._forward(X_batch, y_batch)

        self._backprop(y_hat, y_batch)

        params_and_grads: list[tuple[np.ndarray, np.ndarray]] = []
        for layer in self.layers:
            if layer.trainable:
                params_and_grads.extend(layer.get_params_and_grads())

        self.optimizer._update(params_and_grads)
        return loss

    def add(self, layers: Union[Layer, list[Layer]]):
        if isinstance(layers, Layer):
            self.layers.append(layers)
        elif isinstance(layers, list):
            for layer in layers:
                self.layers.append(layer)
        else:
            raise TypeError("Layer must be a Layer instance or a list of Layer instances.")
    
    def compile(self, optimizer: Optimizer):
        self.optimizer = optimizer

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_build:
            self._build(X.shape[1:])
        x = X
        for layer in self.layers[:-1]: x = layer._forward(x)
        y_hat, _ = self.layers[-1]._forward(x, x)
        return y_hat

    def evaluate(self, X: np.ndarray, y: np.ndarray = None) -> tuple[np.ndarray, float]:
        if not self.is_build:
            self._build(X.shape[1:])

        return self._forward(X, y)
    
    def train(self, train_set: tuple[np.ndarray, np.ndarray] = None,
              test_set: Optional[tuple[np.ndarray, np.ndarray]] = None,
              epochs: int = 1, batch_size: int = 32) -> dict[str, list[float]]:
        
        if train_set is None:
            raise RuntimeError("The train dataset has not been loaded.")
        X_train, y_train = train_set

        test_flag = True
        if test_set is None:
            test_flag = False
            print("WARNING: The test dataset has not been loaded.")
        else:
            X_test, y_test = test_set

        if not self.optimizer:
            raise RuntimeError("The model has not been compiled. Call .compile() before training.")
        
        if not self.is_build:
            self._build(X_train.shape[1:])
            self.is_build = True
        
        n_samples = X_train.shape[0]
        indices = np.arange(n_samples)

        history = {
            "train_loss": [],
            "test_loss": [],
            }

        with tqdm(range(epochs), desc="Training", leave=True) as pbar:
            for t in pbar:
                epoch_loss = 0.0
                test_loss = 0.0

                np.random.shuffle(indices)
                for i in range(0, n_samples, batch_size):
                    batch_indices = indices[i:i + batch_size]
                    X_batch = X_train[batch_indices]
                    y_batch = y_train[batch_indices]

                    epoch_loss += self._batch_update(X_batch, y_batch)

                avg_loss = epoch_loss / batch_size
                history["train_loss"].append(avg_loss)

                if test_flag:
                    y_hat, test_loss = self.evaluate(X_test, y_test)
                    history["test_loss"].append(test_loss)
                    history["y_pred"] = y_hat

                pbar.set_postfix(train_loss=avg_loss, test_loss=test_loss)

        return history