import numpy as np

from loss.base import Loss


class NLLL(Loss):
    def __init__(self):
        super(NLLL, self).__init__()

    def forward(self, x: np.array, y: np.array) -> np.array:
        """
        Computes the negative log-likelihood loss for probabilistic x and one-hot y
        input x: batch x dim x 1
        input y: batch x 1
        return:  batch x 1
        """

        l = np.log(x[np.arange(y.shape[0]), y.flatten()])
        l = l.reshape(-1, 1)
        if self._compute_grad:
            self.grad = self._get_grad(x, y)

        return l

    def _get_grad(self, x: np.array, y: np.array) -> np.array:
        loc = np.arange(y.shape[0])

        # Get un-flattened gradient
        grad = x[loc, y.flatten()]
        grad = -1/grad

        # Error signal
        d = np.zeros_like(x)
        d[loc, y.flatten()] = grad
        return d

    def backward(self) -> np.array:
        return self.grad


class BCE(Loss):
    def __init__(self):
        super(BCE, self).__init__()

    def forward(self, x: np.array, y: np.array) -> np.array:
        """
        Computes the negative log-likelihood loss for probabilistic x and one-hot y
        input x: batch x 1 x 1
        input y: batch x 1
        return:  batch x 1
        """

        # Simplifies computations below
        x = x.squeeze(-1)

        l = - (y * np.log(x) + (1 - y) * np.log(1 - x))
        if self._compute_grad:
            self.grad = self._get_grad(x, y)

        return l

    def _get_grad(self, x: np.array, y: np.array) -> np.array:
        d = (x - y)/(x * (1 - x))
        return np.expand_dims(d, axis=-1)

    def backward(self) -> np.array:
        return self.grad