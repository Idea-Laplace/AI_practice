import numpy as np
from .core_classes import *

# Subclasses of 'Function' ----------------------------------------------
class Add(Function):
    def forward(self, *xs) -> np.ndarray:
        xs = [as_array(x) for x in xs]
        y = sum(xs)

        return y

    def backward(self, grad_y: np.ndarray) -> np.ndarray:
        grad_xs = (grad_y,) * len(self.inputs)

        return grad_xs


class Multiply(Function):
    def forward(self, *xs) -> np.ndarray:
        xs = [as_array(x) for x in xs]
        y = None
        for x in xs:
            if y is None:
                # The copy() method is essential since then input should not be changed.
                y = x.copy()
            else:
                y *= x
        return y

    def backward(self, grad_y: np.ndarray) -> np.ndarray:
        xs = [input.data for input in self.inputs]

        grad_xs = []
        for i in range(len(xs)):
            # list slicing is relatively robust to index range
            temp = xs[:i] + xs[i+1:]
            grad_x = None
            for x_r in temp:
                if grad_x is None:
                    grad_x = x_r.copy()
                else:
                    grad_x *= x_r
            grad_xs.append(grad_x * grad_y)

        return tuple(grad_xs)

class Negation(Function):
    def forward(self, x):
        return -x
    
    def backward(self, grad_y):
        return -grad_y

class Subtraction(Function):
    def forward(self, x0, x1):
        # for rsub
        x1 = as_array(x1)
        y = x0 - x1
        return y
    
    def backward(self, grad_y):
        return grad_y, -grad_y

class Division(Function):
    def forward(self, x0, x1):
        # for rdiv
        x1 = as_array(x1)
        # Division zero error is already implemented in the numpy module.
        y = x0 / x1
        return y
    
    def backward(self, grad_y):
        x0, x1 = self.inputs[0].data, self.inputs[1].data
        grad_x0 = grad_y / x1
        grad_x1 = grad_y * (-x0 / x1 ** 2)
        return grad_x0, grad_x1


class Power(Function):
    def __init__(self, c):
        self.c = c
    
    def forward(self, x):
        y = x ** self.c
        return y
    
    def backward(self, grad_y):
        x = self.inputs[0].data
        c = self.c
        grad_x = grad_y * c * x ** (c - 1)
        return grad_x


# Supportive functions --------------------------------------------

'''
When a numpy array of zero-dimension go through a numpy method,
there would be a chance that the the type of output instance is
just a scalar type, not a numpy array, which is desired.
To avoid this, the following as_array function checks whether the
output is scalar or not, and converts it to a numpy array if necessary.
'''
def add(*xs: Variable) -> Variable:
    return Add()(*xs)

def mul(*xs: Variable) -> Variable:
    return Multiply()(*xs)

def neg(x: Variable) -> Variable:
    return Negation()(x)

def sub(x0, x1) -> Variable:
    return Subtraction()(x0, x1)

def rsub(x0, x1) -> Variable:
    return Subtraction()(x1, x0)

def div(x0, x1) -> Variable:
    return Division()(x0, x1)

def rdiv(x0, x1) -> Variable:
    return Division()(x1, x0)

def pow(x, c) -> Variable:
    return Power(c)(x)

def numerical_gradient(f: Function, *xs: Variable):
    eps = 1e-5
    # Use ravel(), not flatten(), as ravel() indicates the same instance but view it as flat.
    # Even if the x is of ndim zero, the ravel automatically cast x_flatten into ndim 1
    grads = []
    for x in xs:
        x_flatten = x.data.ravel()
        grad_x = np.zeros_like(x_flatten)

    # The algorithm of this 'numerical_gradient' refers to the 'centered divided difference' method.
    # ,in which the formula is represented by f'(x) = lim_(h->0)[(f(x+h) - f(x-h)) / (2*h)]
        for i in range(x_flatten.size):
            x_flatten[i] += eps
            fxr = f(*xs)
            
            x_flatten[i] -= 2 * eps
            fxl = f(*xs)

            grad_x[i] = (fxr.data - fxl.data) / (2 * eps) 
    
        grad_x = grad_x.reshape(x.data.shape)
        grads.append(grad_x)

    return tuple(grads)
