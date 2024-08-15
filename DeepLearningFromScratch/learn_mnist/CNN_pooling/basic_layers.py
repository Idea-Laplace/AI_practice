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
    def __init__(self, w_size: tuple, b_size: int):
        self.w = np.random.normal(0, np.sqrt(2 / w_size[0]), w_size)
        self.b = np.zeros(b_size)
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


class Normalizer:
    def __init__(self, epsilon: float=1e-8):
        self.x = None
        self.y = None
        self.mean = None
        self.var = None
        self.epsilon = epsilon

    
    def forward(self, x: np.ndarray) -> np.ndarray:
        self.x = x
        self.mean = x.mean(axis=0)
        self.var = x.var(axis=0)
        norm_x = (x - self.mean) / np.sqrt(self.var + self.epsilon)
        self.y = norm_x

        return norm_x
    
    def backward(self, dout):
        '''
        x: input
        y: out
        n: number of elements in x
        m: mean
        v: var
        s: stddev
        I: identity matrix

        sdy = dx(I - (1/n)(ones square matrix) - yy^T)
        dx = sdy(I - (1/n)(ones square matrix) - yy^T)

        '''
        restore_shape = self.x.shape
        s = np.sqrt(self.var + self.epsilon).flatten()
        n = self.x.shape[0]
        dout = dout.reshape(dout.shape[0], -1)
        self.y = self.y.reshape(self.y.shape[0], -1)
        self.x = self.x.reshape(self.x.shape[0], -1)
        # Identity matrix
        mat1 = np.eye(dout.shape[1]) / s   
        # 1/n * (n by n) square matix of 1's
        mat2 = np.ones((dout.shape[1], dout.shape[1])) / (n * s)
        # Symmetrical matrix, y^T y
        mat3 = np.dot(self.y.T, self.y) / (n * s)
        dx = np.dot(dout, mat1 - mat2 - mat3)
        dx = dx.reshape(*restore_shape)

        return dx






class DropOut:
    def __init__(self, ratio=0.15):
        self.dropout_ratio = ratio
        self.mask = None
    
    def forward(self, x, train_flg=True):
        if train_flg:
            self.mask = np.random.rand(*x.shape) > self.dropout_ratio
            return x * self.mask
        else:
            return x * (1 - self.dropout_ratio)
    
    def backward(self, dout):
        return dout * self.mask

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

def numerical_gradient(f , x: np.ndarray) -> np.ndarray:
     h = 1e-4
     x_flat = x.ravel()
     grad = np.zeros_like(x)
     grad_flat = grad.ravel()

     if not x.flags['C_CONTIGUOUS']:
         raise ValueError('Input array must be contiguous in memory.')

     for i in range(x_flat.size):
         temp = x_flat[i]

         x_flat[i] += h
         fxh_r = f(x)

         x_flat[i] -= 2 * h
         fxh_l = f(x)

         grad_flat[i] = (fxh_r - fxh_l) / (2 * h)

         x_flat[i] = temp

     return grad

if __name__ == '__main__':
    x = np.random.randn(4, 4, 4) * 10
    test = Normalizer()
    norm_x = test.forward(x)
    print(x)
    print(norm_x)
    print(norm_x.mean())
    print(norm_x.var())
    temp_func = lambda x: np.sum(test.forward(x))
    nu_grad = numerical_gradient(temp_func, x) * (0.01 * x)
    bp_grad = test.backward(nu_grad) 
    print(f"bp_grad(dx) :\n {bp_grad}")
    print(f"0.01x(dx):\n {0.01 * x}")
    print(f"ratio:\n {nu_grad / bp_grad}")