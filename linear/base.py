import numpy as np


class Layer(object):
    def __init__(self, dim_x, dim_z, backprop = True):
        # If in training mode, calculate gradients
        self._compute_grad = True
        self.has_params = True

        # Type of update, using backprop or gradient free update
        self._backprop = backprop

        # Dimensions of mapping
        self.dim_x = dim_x
        self.dim_z = dim_z

        # Initialise parameters
        self.W = np.random.randn(dim_z, dim_x) / np.sqrt(dim_x)
        self.b = np.random.randn(dim_z, 1) / np.sqrt(dim_x)

        # Place holders for updating parameters
        self.W_update = np.zeros((dim_z, dim_x))
        self.b_update = np.zeros((dim_z, 1))

        # Save input
        self.x = None

        # Type of backpropagation used, gradient or gradient-free
        self.EBP = self.W.T
        if not backprop:
            self.EBP = np.random.randn(dim_x, dim_z)/np.sqrt(dim_x)

    def train(self):
        self._compute_grad = True

    def eval(self):
        self._compute_grad = False

    def __call__(self, x: np.array, *args, **kwargs) -> np.array:
        # Error check, input to activation must have 3 dims (batch x dim x 1)
        if len(x.shape) != 3:
            raise ValueError("Input must have 3 dimensions, input has shape", x.shape)

        # Error check, input to activation must be a column vector (batch x dim x 1)
        if x.shape[-1] != 1:
            raise ValueError("Input must be a column vector, input has shape", x.shape)

        return self.forward(x)

    def forward(self, x: np.array) -> np.array:
        pass

    def _get_grad(self, x: np.array, z: np.array) -> np.array:
        pass

    def get_params_update(self, x: np.array, d: np.array):
        pass

    def backward(self, d: np.array) -> np.array:
        pass