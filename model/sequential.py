from typing import List
import numpy as np


from activation.base import Activation
from linear.linear import Linear
from loss.base import Loss


class Sequential(object):
    def __init__(self, sizes: np.array, activations: List[Activation], loss = Loss, backprop = True):
        assert len(sizes)-1 == len(activations)

        self.layerflow = []
        for i in range(len(sizes) - 1):
            self.layerflow.append(Linear(sizes[i], sizes[i+1], backprop = backprop))
            self.layerflow.append(activations[i]())

        self.loss = loss()

    def train(self):
        for layer in self.layerflow:
            layer._compute_grad = True
        self.loss._compute_grad = True

    def eval(self):
        for layer in self.layerflow:
            layer._compute_grad = False
        self.loss._compute_grad = False

    def __call__(self, x: np.array, *args, **kwargs) -> np.array:
        return self.forward(x)

    def forward(self, x: np.array) -> np.array:
        for comp in self.layerflow:
            x = comp(x)

        return x

    def eval_loss(self, x: np.array, y: np.array) -> np.array:
        return self.loss(x, y)

    def backward(self):
        # Obtain the first error signal and now backpropagate
        d = self.loss.backward()

        for comp in self.layerflow[::-1]:

            # Compute update metrics for units with parameters
            if comp.has_params:
                comp.get_params_update(d)

            # Signal is fed backwards
            d = comp.backward(d)

    def update(self, lr = 1e-2):
        for comp in self.layerflow:
            if comp.has_params:
                comp.update(lr)