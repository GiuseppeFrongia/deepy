from abc import ABC, abstractmethod
from typing import Optional
import numpy as np

from ..loss_func import LossFunc
from ..activations import Activation, Linear

class Layer(ABC):
    def __init__(self):
        self.weights: Optional[np.ndarray] = None
        self.bias: Optional[np.ndarray] = None
        self.grad_weights: Optional[np.ndarray] = None
        self.grad_bias: Optional[np.ndarray] = None

        self.output_shape: Optional[tuple[int, ...]] = None

        self.trainable: bool = True
        self.last_input: Optional[np.ndarray] = None
        self.last_actv: Optional[np.ndarray] = None

    @abstractmethod
    def _build(self, input_shape: tuple[int, ...]): pass

    @abstractmethod
    def _forward(self, *args) -> np.ndarray: pass

    @abstractmethod
    def _backprop(self, *args) -> np.ndarray: pass

    def get_params_and_grads(self) -> list[np.ndarray]:
        return [(self.weights, self.grad_weights)] + [(self.bias, self.grad_bias)]
    
    def count_params(self) -> int:
        params = 0
        if self.weights is not None:
            params += self.weights.size
        if self.bias is not None:
            params += self.bias.size
        return params
    

class Input(Layer):
    def __init__(self, shape: tuple[int, ...]):
        super().__init__()
        self.output_shape = shape
        self.trainable = False

    def _build(self, input_shape: tuple[int, ...]):
        if input_shape != self.output_shape:
            raise ValueError(f"The number of features in the dataset ({input_shape}) differs from the model's expected input dimension ({self.output_shape}).")
        
    def _forward(self, inputs: np.ndarray) -> np.ndarray: return inputs

    def _backprop(self, grad): return grad

    def count_params(self) -> int: return 0


class Dense(Layer):
    def __init__(self, shape: int,
                 activation: Activation=Linear):
        super().__init__()
        self.shape: int = shape
        self.output_shape = (self.shape,)
        self.activation: Activation = activation()

    def _build(self, input_shape: tuple[int, ...]):
        if len(input_shape) != 1:
            raise ValueError(f"Dense layer expected input_shape to be 1D (features,), but got {input_shape}.")
        
        input_features = input_shape[0]

        #inizializzazione He dei pesi (ideale per ReLU)
        limit = np.sqrt(2. / input_features)
        self.weights = np.random.randn(input_features, self.shape) * limit
        self.bias = np.zeros((self.shape,))

    def _forward(self, inputs: np.ndarray) -> np.ndarray:
        self.last_actv = inputs
        self.last_input = inputs @ self.weights + self.bias
        return self.activation.func(self.last_input)

    def _backprop(self, grad: np.ndarray) -> np.ndarray:
        grad = grad * self.activation.deriv(self.last_input)
        self.grad_weights = self.last_actv.T @ grad
        self.grad_bias = np.sum(grad, axis=0)

        return grad @ self.weights.T


class Output(Dense):
    def __init__(self, loss_func: Optional[LossFunc] = None,
                 shape: int=1,
                 activation: Activation=Linear):
        super().__init__(shape, activation)
        self.loss_func: LossFunc = loss_func()
    
    def _forward(self, inputs, y_batch):
        y_hat = super()._forward(inputs)
        loss = self.loss_func._loss(y_hat=y_hat, targets=y_batch)
        return y_hat, loss

    def _backprop(self, y_hat, y_batch):
        grad = self.loss_func._grad(y_hat, y_batch)
        return super()._backprop(grad)