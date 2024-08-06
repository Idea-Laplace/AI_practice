import numpy as np


# Activation functions: step, sigmoid, relu
class ReLu:
    def __init__(self):
        # Stores information for the input array in which element is lower than 0.
        self.mask = None

    def forward(self, x: np.ndarray):
        self.mask = (x <= 0)
        # To protect the real input array
        out = x.copy()
        out[self.mask] = 0

        return out

    def backward(self, dout):
        dout[self.mask] = 0

        return dout

class Sigmoid:
    def __init__(self):
        self.out = None

    def forward(self, x):
        out = 1 / (1 + np.exp(-x))
        self.out = out
        return out
    '''
    y = 1 / (1 + e^-x)
    [1 / (1 + e^-x)]` = -(-e^-x)/(1 + e^-x)^2
    = ((1 + e^-x) - 1)/(1 + e^-x)^2
    = ((1/y) - 1)*y^2
    = y(1 - y)
    '''
    def backward(self, dout):
        dx = dout * (1 - self.out) * self.out
        return dx

class Affine:
    def __init__(self, w, b):
        self.w = w
        self.b = b
        self.x = None
        self.dw = None
        self.db = None

    def forward(self, x: np.ndarray):
        self.x = x
        out = np.dot(x, self.w) + self.b

        return out

    def backward(self, dout):
        dx = np.dot(dout, self.w.T)
        self.dw = np.dot(self.x.T, dout)
        # np.sum is used as the b is broadcasted.
        self.db = np.sum(dout, axis = 0)

        return dx

class SoftmaxLoss:
    def __init__(self):
        self.loss = None
        self.y = None
        self.x = None

    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)
        self.loss = cross_entropy_error(self.y, self.t)
        return self.loss
    

    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        dx = (self.y - self.t) / batch_size

        return dx

# Returns a ndarray containing probabilities of each entity in the input ndarray.
def softmax(x: np.ndarray) -> np.ndarray:
    # To avoid getting crashed value by outliers.
    correction = np.max(x, axis=1).reshape(-1, 1)
    exp_x = np.exp(x - correction)  
    normalizer = exp_x.sum(axis=1).reshape(-1, 1)
    probability = exp_x / normalizer

    return probability

# Error functions
## SSE
def sum_of_squares(y: np.ndarray, t: np.ndarray) -> float:
    return 0.5 * np.sum((y - t) ** 2)

def cross_entropy_error(y: np.ndarray, t: np.ndarray, delta=1e-7) -> float:
    return -np.sum(t * np.log(y + delta))

