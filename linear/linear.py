import numpy as np

from linear.base import Layer


class Linear(Layer):
    def __init__(self, dim_x, dim_z, backprop = True):
        super(Linear, self).__init__(dim_x, dim_z, backprop = backprop)

    def forward(self, x: np.array) -> np.array:
        # Linear mapping
        z = self.W @ x + self.b

        if self._compute_grad:
            if self._backprop:
                self.EBP = self._get_grad(x, z)
            self.x = x

        return z

    def _get_grad(self, x: np.array, z: np.array) -> np.array:
        return self.W.T

    def get_params_update(self, d: np.array):
        if d.shape[1:] != self.b.shape:
            raise ValueError("Error signal must match the shape of the bias. Size of error",
                             d.shape, "and size of bias", self.b.shape)

        self.b_update += np.sum(d, axis = 0)
        self.W_update += np.sum(
            np.array([np.outer(v1, v2) for v1, v2 in zip(d, self.x)]), axis = 0
        )

    def backward(self, d: np.array) -> np.array:
        return self.EBP @ d