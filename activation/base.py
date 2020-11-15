import numpy as np


class Activation(object):
    def __init__(self):
        self._compute_grad = True
        self.grad = None
        self.has_params = False

    def train(self):
        self._compute_grad = True

    def eval(self):
        self._compute_grad = False

    def __call__(self, z: np.array, *args, **kwargs) -> np.array:
        # Error check, input to activation must have 3 dims (batch x dim x 1)
        if len(z.shape) != 3:
            raise ValueError("Input must have 3 dimensions, input has shape", z.shape)

        # Error check, input to activation must be a column vector (batch x dim x 1)
        if z.shape[-1] != 1:
            raise ValueError("Input must be a column vector, input has shape", z.shape)

        return self.forward(z)

    def forward(self, z: np.array) -> np.array:
        pass

    def _get_grad(self, z: np.array, y: np.array) -> np.array:
        pass

    def backward(self, d: np.array) -> np.array:
        pass