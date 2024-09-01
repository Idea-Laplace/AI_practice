import numpy as np
from MyDeZero.core.core_classes import *
from . import core_functions as crf
from MyDeZero.convolution_pooling import convolution as conv
from MyDeZero.cuda import cuda


class Exp(Function):
    def forward(self, x):
        xp = cuda.get_array_module(x)
        y = xp.exp(x)

        return y

    def backward(self, grad_y):
        x = self.inputs[0]
        grad_x = exp(x) * grad_y

        return grad_x


class Log(Function):
    def forward(self, x):
        xp = cuda.get_array_module(x)
        y = xp.log(x)

        return y

    def backward(self, grad_y):
        x = self.inputs[0]
        grad_x = (1 / x) * grad_y

        return grad_x

class Sin(Function):
    def forward(self, x):
        xp = cuda.get_array_module(x)
        y = xp.sin(x)
        return y
    
    def backward(self, grad_y):
        x = self.inputs[0]
        grad_x = grad_y * cos(x)
        return grad_x

class Cos(Function):
    def forward(self, x):
        xp = cuda.get_array_module(x)
        y = xp.cos(x)
        return y
    
    def backward(self, grad_y):
        x = self.inputs[0]
        grad_x = grad_y * (-sin(x))
        return grad_x


class Tanh(Function):
    def forward(self, x):
        xp = cuda.get_array_module(x)
        y = xp.tanh(x)
        return y
    
    #(tanh(x))` = 1 - y**2
    def backward(self, grad_y: Variable) -> Variable:
        y = self.outputs[0]()
        grad_x = grad_y * (1 - y * y)
        return grad_x


class Sigmoid(Function):
    def forward(self, x):
        xp = cuda.get_array_module(x)
        return 1 / (1 + xp.exp(-x))
    
    def backward(self, grad_y: Variable) -> Variable:
        y = self.outputs[0]()
        return grad_y * y * (1 - y)
    

class ReLu(Function):
    def forward(self, x):
        self.mask = (x <= 0)
        y = x.copy()
        y[self.mask] = 0.
        return y
    
    def backward(self, grad_y: Variable) -> Variable:
        arr = grad_y.data.copy()
        arr[self.mask] = 0
        grad_x = Variable(arr)
        return grad_x


class MeanSquareError(Function):
    def forward(self, x0, x1):
        diff = x0 - x1
        y = (diff ** 2).sum() / len(diff)
        return y
    
    def backward(self, grad_y: Variable) -> Variable:
        x0, x1 = self.inputs
        diff = x0 - x1
        grad_x0 = grad_y * diff * (2./ len(diff))
        grad_x1 = -grad_x0

        return grad_x0, grad_x1


class SoftMax(Function):
    def __init__(self, axis=1):
        self.axis = axis

    def forward(self, x):
        xp = cuda.get_array_module(x)
        x_max = x.max(axis=self.axis, keepdims=True)
        y = x - x_max
        y = xp.exp(y)
        y /= y.sum(axis=self.axis, keepdims=True)

        return y
    
    '''
    p_k := exp(x_k) / sum_i(exp(x_i))
    dL/d(x_i) = sum_k((dL/dp_k)(dp_k/dx_i))
    where dp_k / dx_i = p_k(1 - p_k) if i == k else -p_i * p_k

    dL/d(x_i) =  grad_p[i] * p[i] - p[i] * sum_k(grad_p[k] * p[k]) 
    dL/dx = p * grad_p - p * sum(p * grad_p)
    '''
    def backward(self, grad_y: Variable) -> Variable:
        y = self.outputs[0]()
        grad_x = y * grad_y
        sumdx = grad_x.sum(axis=self.axis, keepdims=True)
        grad_x -= y * sumdx
        return grad_x


class CrossEntropy(Function):
    def __init__(self, eps=1e-10):
        self.eps = eps
    
    def forward(self, x, t):
        xp = cuda.get_array_module(x)
        log_x = xp.log(x + self.eps)
        y = -xp.sum(t * log_x) / x.shape[0]
        return y
    
    def backward(self, grad_y: Variable) -> Variable:
        x, t = self.inputs
        grad_x = -grad_y * t / (x * x.shape[0])
        return grad_x


class SoftMaxCrossEntropy(Function):
    def __init__(self, axis=1, eps=1e-15):
        self.axis = axis
        self.eps = eps

    def forward(self, x, t):
        xp = cuda.get_array_module(x)

        x_max = x.max(axis=self.axis, keepdims=True)
        y = x - x_max
        y = xp.exp(y)
        y /= y.sum(axis=self.axis, keepdims=True)
        log_y = xp.log(y + self.eps)
        loss = -xp.sum(t * log_y) / x.shape[0]

        return loss
    
    def backward(self, grad_y: Variable) -> Variable:
        x, t = self.inputs
        p = softmax(x)
        grad_x = grad_y * (p - t) / t.shape[0]
        
        return grad_x 


class DropOut(Function):
    def __init__(self, dropout_ratio=0.5):
        self.dropout_ratio = dropout_ratio
        self.mask = None

    def forward(self, x):
        xp = cuda.get_array_module(x)
        if MyDeZero.Configuration.train:
            self.mask = xp.random.rand(*x.shape) > self.dropout_ratio
            self.scale = xp.array(1.0 - self.dropout_ratio).astype(x.dtype)
            y = (x * self.mask) / self.scale
            return y
        else:
            return x

    def backward(self, grad_y: Variable) -> Variable:
        if MyDeZero.Configuration.train:
            grad_x = grad_y * self.mask / self.scale
            return grad_x
        else:
            return grad_y


class Normal(Function):
    def forward(self, x):
        xp = cuda.get_array_module(x)
        mean = x.sum(axis=0) / x.shape[0]
        center = x - mean
        var = (center ** 2).sum(axis=0) / center.shape[0]
        norm_x = center / xp.sqrt(var + 1e-15)

        self.stddev = xp.sqrt(var + 1e-15)
        return norm_x
    
    def backward(self, grad_y):
        y = self.outputs[0]()
        grad_x = grad_y - mean(y * grad_y) * y - mean(grad_y)
        grad_x /= self.stddev

        return grad_x


class Convolution(Function):
    def __init__(self, kernel_shape: tuple[int], stride: int=1, pad: int=0, block_w=True):
        self.kernel_shape = kernel_shape
        self.stride = stride
        self.pad = pad
        self.block_w = block_w

    def forward(self, x, w, b=None):
        xp = cuda.get_array_module(x)
        if self.block_w:
            OC = w.shape[0]
            self.w2d = w.reshape(w.shape[0], -1).T
        else:
            OC = w.shape[1]
            self.w2d = w

        col_x = conv._im2col(x, self.kernel_shape, self.stride, self.pad)
        self.col_x = col_x
        N, C, H, W = x.shape
        KH, KW = self.kernel_shape
        OH = conv._get_conv_outsize(H, KH, self.stride, self.pad)
        OW = conv._get_conv_outsize(W, KW, self.stride, self.pad)

        y = xp.dot(col_x, self.w2d)
        if b is not None:
            y += b
        y = y.reshape(N, OH, OW, OC).transpose(0, 3, 1, 2)
        return y
    
    def backward(self, grad_y: Variable) -> Variable:
        xp = cuda.get_array_module(grad_y.data)
        x, w, b = self.inputs
        if self.block_w:
            C, KH, KW = w.shape[1:] 
        else:
            KH, KW = self.kernel_shape
            C = x.shape[1]
        N, OC, OH, OW = self.outputs[0]().shape
        grad_y_reshape = grad_y.transpose(0, 2, 3, 1).reshape(N * OH * OW, OC)
        grad_w = xp.dot(self.col_x.T, grad_y_reshape.data)
        grad_x = xp.dot(grad_y_reshape.data, self.w2d.T)

        if self.block_w:
            grad_w = grad_w.T.reshape(OC, C, KH, KW)
        grad_x = conv._col2im(grad_x, x.shape, (KH, KW), self.stride, self.pad)

        grad_x = as_variable(grad_x)
        grad_w = as_variable(grad_w)
        grad_b = None if b.data is None else crf.sum_to(grad_y_reshape, b.shape)
        return grad_x, grad_w, grad_b
        

class Pooling(Function):
    def __init__(self, kernel_shape: tuple[int], stride: int=1, pad: int=0):
        self.kernel_shape = kernel_shape
        self.stride = stride
        self.pad = pad

    def forward(self, x):
        N, C, H, W = x.shape
        KH, KW = self.kernel_shape
        OH = conv._get_conv_outsize(H, KH, self.stride, self.pad)
        OW = conv._get_conv_outsize(W, KW, self.stride, self.pad)
        col_x = conv._im2col(x, self.kernel_shape, self.stride, self.pad)
        blockwise_x = col_x.reshape(-1, KH * KW)
        pooled_x = blockwise_x.max(axis=1, keepdims=True)
        pooled_x = pooled_x.reshape(N, C, OH, OW)

        self.index = blockwise_x.argmax(axis=1)
        return pooled_x
    
    def backward(self, grad_y: Variable) -> Variable:
        xp = cuda.get_array_module(grad_y)
        
        x, = self.inputs
        N, C, H, W = x.shape
        KH, KW = self.kernel_shape
        OH = conv._get_conv_outsize(H, KH, self.stride, self.pad)
        OW = conv._get_conv_outsize(W, KW, self.stride, self.pad)

        flatten_grad_y = grad_y.data.flatten()
        grad_x = xp.zeros((flatten_grad_y.size, KH * KW))

        slice = xp.arange(flatten_grad_y.size), self.index
        grad_x[slice] = flatten_grad_y
        grad_x = grad_x.reshape(N * OH * OW, -1)
        grad_x = conv._col2im(grad_x, x.shape, self.kernel_shape, self.stride, self.pad)
        grad_x = as_variable(grad_x)

        return grad_x
        


#----------------------------------------------------
def exp(x: Variable) -> Variable:
    return Exp()(x)

def log(x: Variable) -> Variable:
    return Log()(x)

def sin(x: Variable) -> Variable:
    return Sin()(x)

def cos(x: Variable) -> Variable:
    return Cos()(x)

def tanh(x: Variable) -> Variable:
    return Tanh()(x)

def sigmoid(x: Variable) -> Variable:
    return Sigmoid()(x)

def relu(x: Variable) -> Variable:
    return ReLu()(x)

def mean(x: Variable, axis: int=0) -> Variable:
    return crf.sum(x, axis) / x.shape[0]

def mse(x0: Variable, x1: Variable) -> Variable:
    return MeanSquareError()(x0, x1)

def softmax(x, axis=1):
    return SoftMax(axis)(x)

def cross_entropy(x, t, eps=1e-10):
    return CrossEntropy(eps)(x, t)

def softmax_cross_entropy(x, t, axis=1):
    return SoftMaxCrossEntropy(axis)(x, t)

def normal(x: Variable)->Variable:
    return Normal()(x)

def convolution(x, w, b, kernel_shape, stride=1, pad=0) -> Variable:
    return Convolution(kernel_shape, stride, pad)(x, w, b)

def pooling(x, kernel_shape, stride: int=1, pad: int=0) -> Variable:
    return Pooling(kernel_shape, stride, pad)(x)

def accuracy(y: Variable, t: Variable):
    y, t = as_variable(y), as_variable(t)

    pred = y.data.argmax(axis=1)
    ans = t.data.argmax(axis=1)
    result = (pred == ans)
    acc = result.mean()

    return as_variable(as_array(acc))

#-----------------------------------------------------