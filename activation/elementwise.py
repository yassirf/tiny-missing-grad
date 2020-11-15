import numpy as np

from activation.base import Activation


class Identity(Activation):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, z: np.array) -> np.array:

        y = z
        if self._compute_grad:
            self.grad = self._get_grad(z, y)

        return y

    def _get_grad(self, z: np.array, y: np.array) -> np.array:
        grad = np.array([np.eye(v.shape[0]) for v in z])
        return grad

    def backward(self, d: np.array) -> np.array:
        return d


class ReLU(Activation):
    def __init__(self):
        super(ReLU, self).__init__()

    def forward(self, z: np.array) -> np.array:

        y = z * (z > 0)
        if self._compute_grad:
            self.grad = self._get_grad(z, y)

        return y

    def _get_grad(self, z: np.array, y: np.array) -> np.array:
        grad = np.array([np.diag(v.flatten() > 0) for v in z])
        return grad

    def backward(self, d: np.array) -> np.array:
        return self.grad @ d


class Sigmoid(Activation):
    def __init__(self):
        super(Sigmoid, self).__init__()

    def forward(self, z: np.array) -> np.array:
        mask = z > 0

        # This represents a more numerically stable version of sigmoid,
        y = np.where(
            mask,                           # Where mask is true
            1/(1 + np.exp(-z)),             # Evaluate sigmoid using this
            np.exp(z)/(np.exp(z) + 1)       # If not use this
        )

        if self._compute_grad:
            self.grad = self._get_grad(z, y)

        return y

    def _get_grad(self, z: np.array, y: np.array) -> np.array:
        grad = y * (1 - y)
        grad = np.array([np.diag(v) for v in grad])
        return grad

    def backward(self, d: np.array) -> np.array:
        return self.grad @ d