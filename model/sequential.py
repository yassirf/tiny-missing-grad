from typing import List
import numpy as np


from activation.base import Activation
from linear.linear import Linear


class Sequential(object):
    def __init__(self, sizes: np.array, activations: List[Activation], backprop = True):
        assert len(sizes)-1 == len(activations)

        self.layerflow = []
        for i in range(len(sizes) - 1):
            self.layerflow.append(Linear(sizes[i], sizes[i+1], backprop = backprop))
            self.layerflow.append(activations[i]())

    def train(self):
        for layer in self.layerflow:
            layer._compute_grad = True

    def eval(self):
        for layer in self.layerflow:
            layer._compute_grad = False

    def __call__(self, x: np.array, *args, **kwargs) -> np.array:
        return self.forward(x)

    def forward(self, x: np.array) -> np.array:
        for comp in self.layerflow:
            x = comp(x)

        return x

