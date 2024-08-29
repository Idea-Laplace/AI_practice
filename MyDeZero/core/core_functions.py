import numpy as np
from MyDeZero.core.core_classes import *

# Subclasses of 'Function' ----------------------------------------------
class Add(Function):
    def forward(self, x0, x1) -> np.ndarray:
        x1 = as_array(x1)
        y  = x0 + x1
        return y

    def backward(self, grad_y: np.ndarray) -> np.ndarray:
        return grad_y, grad_y


class Multiply(Function):
    def forward(self, x0, x1) -> np.ndarray:
        x1 = as_array(x1)
        y = x0 * x1
        return y

    def backward(self, grad_y):
        x0, x1 = self.inputs
        return grad_y * x1, grad_y * x0

class Negation(Function):
    def forward(self, x):
        return -x
    
    def backward(self, grad_y):
        return -grad_y

class Subtraction(Function):
    def forward(self, x0, x1):
        x1 = as_array(x1)
        y = x0 - x1
        return y
    
    def backward(self, grad_y):
        return grad_y, -grad_y

class Division(Function):
    def forward(self, x0, x1):
        x1 = as_array(x1)
        # Division zero error is already implemented in the numpy module.
        y = x0 / x1
        return y
    
    def backward(self, grad_y):
        x0, x1 = self.inputs
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
        x = self.inputs[0]
        c = self.c
        grad_x =  c * x ** (c - 1) * grad_y
        return grad_x


class Reshape(Function):
    def __init__(self, shape):
        self.shape = shape

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.x_shape = x.shape
        y = x.reshape(self.shape)
        return y

    def backward(self, grad_y: Variable) -> Variable:
        return reshape(grad_y, self.x_shape)


class Transpose(Function):
    def __init__(self, axes=None):
        self.axes = axes

    def forward(self, x: np.ndarray) -> np.ndarray:
        y = x.transpose(self.axes)
        return y
    
    def backward(self, grad_y: Variable) -> Variable:
        if self.axes is None:
            return transpose(grad_y)
        else:
            axes_len = len(self.axes)
            # The part '% axes_len' is for when negative index is used.
            # The argsort method returns an np.ndarray, 
            inv_axes = tuple(np.argsort([ax % axes_len for ax in self.axes]))
            return transpose(grad_y, inv_axes)


class Sum(Function):
    def __init__(self, axis, keepdims: bool):
        self.axis = axis
        self.keepdims = keepdims

    def forward(self, x:np.ndarray) -> np.ndarray:
        self.x_shape = x.shape
        return x.sum(axis=self.axis, keepdims=self.keepdims)
    
    def backward(self, grad_y: Variable) -> Variable:
        grad_y = reshape_sum_backward(grad_y, self.x_shape, self.axis,\
                                      self.keepdims)
        grad_x = broadcast_to(grad_y, self.x_shape)
        return grad_x

class BroadcastTo(Function):
    def __init__(self, shape):
        self.shape = shape
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        self.x_shape = x.shape
        y = np.broadcast_to(x, self.shape)
        return y
    
    def backward(self, grad_y: Variable) -> Variable:
        grad_x = sum_to(grad_y, self.x_shape)
        return grad_x


# Do not confuse with 'Sum' or 'Add'
class SumTo(Function):
    def __init__(self, shape):
        self.shape = shape

    def forward(self, x):
        self.x_shape = x.shape
        ndim = len(self.shape)
        # ndim should be not larger than x.ndim
        lead = x.ndim - ndim
        # if lead = 0, lead_axis is (), an empty tuple
        # axes that outdimension that of self.shape
        lead_axis = tuple(range(lead))

        # 'lead'is added to each element of 'axis' to make sure that
        # the lead_axis is followed by 'axis' in x
        # When sx == 1, the broadcasting would have been done on x
        axis = tuple([i + lead for i, sx in enumerate(self.shape) if sx == 1])
        y = x.sum(lead_axis + axis, keepdims=True)
        if lead > 0:
            # Since keepdims remains dummy dimensions, this is needed
            y = y.squeeze(lead_axis)
        return y   
    
    def backward(self, grad_y):
        grad_x = broadcast_to(grad_y, self.x_shape)
        return grad_x


class MatrixMul(Function):
    def forward(self, x: np.ndarray, W: np.ndarray) -> np.ndarray:
        y = x.dot(W)
        return y
    
    def backward(self, grad_y: Variable) -> Variable:
        if grad_y.ndim == 1:
            wrap_grad_y = grad_y.reshape(1, -1)
        else:
            wrap_grad_y = grad_y

        x, W = self.inputs
        if W.ndim == 1:
            WT = W.reshape(1, -1).T
        else:
            WT = W.T

        if x.ndim == 1:
            xT = x.reshape(1, -1).T
        else:
            xT = x.T

        grad_x = mat_mul(wrap_grad_y, WT)
        grad_W = mat_mul(xT, wrap_grad_y)
        return grad_x, grad_W


class LinearTransfrom(Function):
    def forward(self, x, W, b=None):
        self.x = x
        self.W = W
        
        y = np.dot(x, W)
        if b is not None:
            y += b

        return y

    def backward(self, grad_y: Variable) -> Variable:
        x, W, b = self.inputs
        grad_x = mat_mul(grad_y, W.T)
        grad_W = mat_mul(x.T, grad_y)
        grad_b = None if b.data is None else sum_to(grad_y, b.shape)

        return grad_x, grad_W, grad_b


class GetItem(Function):
    def __init__(self, slices: np.ndarray):
        self.slices = slices
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        y = x[self.slices]
        return y
    
    def backward(self, grad_y: Variable) -> Variable:
        x, = self.inputs
        func = GetItemGrad(self.slices, x.shape)
        return func(grad_y)


class GetItemGrad(Function):
    def __init__(self, slices, in_shape):
        self.slices = slices
        self.in_shape = in_shape
    
    def forward(self, grad_ydata: np.ndarray) -> np.ndarray:
        # Restore the shape of an input
        grad_xdata = np.zeros(self.in_shape)
        # Sum would accumulated if slices called an index duplicatively.
        # Number of elements in grad_ydata should be same with the number of slices.
        # elements in grad_ydata are called just the way that an iteration calls
        np.add.at(grad_xdata, self.slices, grad_ydata)
        return grad_xdata
    
    def backward(self, grad_grad_x: Variable) -> Variable:
        return get_item(grad_grad_x, self.slices)




# Supportive functions --------------------------------------------

'''
When a numpy array of zero-dimension go through a numpy method,
there would be a chance that the the type of output instance is
just a scalar type, not a numpy array, which is desired.
To avoid this, the following as_array function checks whether the
output is scalar or not, and converts it to a numpy array if necessary.
'''
def add(x0: Variable, x1: Variable) -> Variable:
    return Add()(x0, x1)

def mul(x0: Variable, x1:Variable) -> Variable:
    return Multiply()(x0, x1)

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

def reshape(x, shape) -> Variable:
    if x.shape == shape:
        return as_variable(x)
    return Reshape(shape)(x)

def transpose(x: Variable, axes=None) -> Variable:
    return Transpose(axes)(x)

def sum(x: Variable, axis=None, keepdims=False) -> Variable:
    return Sum(axis, keepdims)(x)

def broadcast_to(x: Variable, shape):
    if x.shape == shape:
        return as_variable(x)
    return BroadcastTo(shape)(x)

def sum_to(x, shape):
    if x.shape == shape:
        return as_variable(x)
    return SumTo(shape)(x)

def mat_mul(x, W):
    return MatrixMul()(x, W)

def linear_transform(x, W, b=None):
    return LinearTransfrom()(x, W, b)

def get_item(x, slices):
    return GetItem(slices)(x)

def numerical_gradient(f: Function, *xs: Variable, eps=1e-5):
    # Use ravel(), not flatten(), as ravel() indicates the same instance but view it as flat.
    # Even if the x is of ndim zero, the ravel automatically cast x_flatten into ndim 1
    grads = []
    for x in xs:
        x_flatten = x.data.ravel()
        grad_x = np.zeros_like(x_flatten)

    # The algorithm of this 'numerical_gradient' refers to the 'centered divided difference' method.
    # ,in which the formula is represented by f'(x) = lim_(h->0)[(f(x+h) - f(x-h)) / (2*h)]
        for i in range(x_flatten.size):
            temp = x_flatten[i]
            x_flatten[i] += eps
            fxr = f(*xs)
            
            x_flatten[i] -= 2 * eps
            fxl = f(*xs)

            grad_x[i] = (fxr.data - fxl.data) / (2 * eps) 
            x_flatten[i] = temp
    
        grad_x = grad_x.reshape(x.data.shape)
        grads.append(grad_x)

    return tuple(grads) if len(grads) > 1 else grads[0]

def reshape_sum_backward(grad_y, x_shape, axis, keepdims):
    ndim = len(x_shape)
    # Make the type of axis consistent with tuple---
    tupled_axis = axis
    if axis is None:
        tupled_axis = None
    elif not isinstance(axis, tuple):
        tupled_axis = (axis,)
    #\-----------------------------------------------

    # When the 'keepdims' is concerned.
    # Even though keepdims is false, that would be meaningless
    # if ndim is 0(which make the sum mere scalar)
    # or if axis is None(ditto.)
    if not (ndim == 0 or tupled_axis is None or keepdims):
        # Make 'negative' axis into positive.
        actual_axis = [a if a >= 0 else a + ndim for a in tupled_axis]
        shape = list(grad_y.shape)
        # Before broadcasting, restore dimensions as if the sum invoked with 'keepdims=True'
        for a in sorted(actual_axis):
            shape.insert(a, 1)
    else:
        shape = grad_y.shape

    grad_y = grad_y.reshape(shape)  # reshape
    return grad_y

def dropout(x, dropout_ratio=0.5):
    x = as_variable(x)

    if Configuration.train:
        mask = np.random.rand(*x.shape) > dropout_ratio
        scale = np.array(1.0 - dropout_ratio).astype(x.dtype)
        y = mask / scale
        return y
    else:
        return x