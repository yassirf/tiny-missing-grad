import numpy as np

from activation.base import Activation


class Softmax(Activation):
    def __init__(self):
        super(Softmax, self).__init__()

    def forward(self, z: np.array) -> np.array:
        z = z - np.max(z, axis = 1, keepdims = True)

        y = np.exp(z)
        y = y/np.sum(y, axis = 1, keepdims = True)

        if self._compute_grad:
            self.grad = self._get_grad(z, y)

        return y

    def _get_grad(self, z: np.array, y: np.array) -> np.array:
        grad = -np.array([np.outer(v, v) for v in y])
        grad += np.array([np.diag(v.flatten()) for v in y])
        return grad

    def backward(self, d: np.array) -> np.array:
        return self.grad @ d