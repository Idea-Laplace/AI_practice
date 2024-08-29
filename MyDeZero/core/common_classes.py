import numpy as np
from MyDeZero import Layer, Parameter, Variable, Optimizer
import MyDeZero.utilities.utils as utils
import MyDeZero.core.common_functions as cmf
import MyDeZero.core.core_functions as crf
import MyDeZero.convolution_pooling.convolution as conv


# Layer--------------------------------------------------------------------
class Linear(Layer):
    def __init__(self, output_size: int,\
                 input_size=None,\
                 nobias: bool=False,\
                 dtype=np.float32):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.dtype = dtype

        self.W = Parameter(None, name='W')
        if self.input_size is not None:
            self._init_W()
        
        if nobias:
            self.b = None
        else:
            self.b = Parameter(np.zeros(output_size, dtype=dtype), name='b')
    
    def _init_W(self):
        I, O = self.input_size, self.output_size
        W_data = np.random.randn(I, O).astype(self.dtype) * np.sqrt(1 / I)
        self.W.data = W_data
    
    def forward(self, x):
        if self.W.data is None:
            if x.ndim == 1:
                x = x.reshape(1, -1)
            # Number of columns of x == number of rows of self.W.data
            self.input_size = x.shape[1]
            self._init_W()
        
        y = crf.linear_transform(x, self.W, self.b)
        return y


class Conv2d(Layer):
    def __init__(self, out_channels, kernel_shape, stride=1,\
                pad=0, nobias=False, dtype=np.float32, in_channels=None):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_shape = kernel_shape
        self.stride = stride
        self.pad = pad
        self.dtype = dtype

        self.W = Parameter(None, name='W')
        if in_channels is not None:
            self._init_W()
        
        if nobias:
            self.b = None
        else:
            self.b = Parameter(np.zeros(out_channels, dtype=dtype), name='b')
        
    def _init_W(self):
        C, OC = self.in_channels, self.out_channels
        KH, KW = self.kernel_shape
        scale = np.sqrt(1 / (C * KH * KW))
        W_data = scale * np.random.randn(OC, C, KH, KW).astype(self.dtype)
        self.W.data = W_data
    
    def forward(self, x):
        if self.W.data is None:
            self.in_channels = x.shape[1]
            self._init_W()
        KH, KW = self.W.shape[2:]
        y = cmf.Convolution((KH, KW), self.stride, self.pad)(x, self.W, self.b)

        return y
            


class Model(Layer):
    def plot(self, *inputs, to_file='model.png'):
        y = self.forward(*inputs)
        return utils.plot_dot_graph(y, verbose=True, to_file=to_file)


# MLP stands for 'Multi-Layer-Perceptron'
class MLP(Layer):
    def __init__(self, fc_output_size: tuple[int],\
                activation=cmf.sigmoid, normal: bool=True, dropout=None):
        super().__init__()
        self.activation = activation
        self.normal = normal
        self.dropout = dropout
        self.layers = []

        for i, output_size in enumerate(fc_output_size):
            layer = Linear(output_size)
            setattr(self, 'linear' + str(i), layer)
            self.layers.append(layer)
        
    def forward(self, x):
        # The last layer does not undergo the activation function.
        for layer in self.layers[:-1]:
            if self.normal:
                x = cmf.normal(layer(x))
            else:
                x = layer(x)
            x = self.activation(x)

            if self.dropout is not None:
                x = cmf.DropOut(self.dropout)(x)
        return cmf.normal(self.layers[-1](x)) if self.normal else self.layers[-1](x)


class CNN(Layer):
    def __init__(self, channels: tuple[int],\
                kernel_shape: tuple[int],\
                stride: int=1, pad: int=0,\
                pooling: bool=True,\
                in_channels=None,\
                activation=cmf.relu):

        super().__init__()
        self.stride = stride
        self.pad = pad
        self.kernel_shape = kernel_shape
        self.pooling = pooling
        self.activation = activation
        self.layers = []
        for i, channel in enumerate(channels):
            layer = Conv2d(channel, kernel_shape, stride, pad, in_channels)
            setattr(self, 'Conv' + str(i), layer)
            self.layers.append(layer)
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
            #print(x.generation, type(x))
            x = self.activation(x)
            #print(x.generation, type(x))

        if self.pooling:
            x = cmf.pooling(x, self.kernel_shape, self.stride, self.pad)
        return x


class CNNMLP(Layer):
    def __init__(self, cnns: tuple[CNN], mlp: MLP):
        super().__init__()
        self.cnn_layers = []
        self.mlp_layer = mlp

        for i, cnn in enumerate(cnns):
            setattr(self, 'cnn' + str(i), cnn)
            self.cnn_layers.append(cnn)

    def forward(self, x):
        for cnn in self.cnn_layers:
            x = cnn(x)
        N, C, H, W = x.shape
        x = x.reshape(N, -1)
        x = self.mlp_layer(x)
        return x

# Opimizer-------------------------------------------------------------------
# SGD: Stochastic Gradient Descent
class SGD(Optimizer):
    def __init__(self, lr=0.01):
        super().__init__()
        self.lr = lr
    
    def update_one(self, param):
        param.data -= self.lr * param.grad.data


class MomentumSGD(Optimizer):
    def __init__(self, lr=0.01, momentum=0.9):
        super().__init__()
        self.lr = lr
        self.momentum = momentum
        self.velocity_dict = {}
    
    def update_one(self, param):
        v_key = id(param)
        if v_key not in self.velocity_dict:
            self.velocity_dict[v_key] = np.zeros_like(param.data)
        
        v = self.velocity_dict[v_key]
        # The momentum functions like friction on an incline
        v *= self.momentum
        v -= self.lr * param.grad.data
        param.data += v


class AdaGrads(Optimizer):
    def __init__(self, lr=0.01, eps=1e-7):
        super().__init__()
        self.lr = lr
        self.eps = eps
        self.h_dict = {}
    
    def update_one(self, param):
        h_key = id(param)
        if h_key not in self.h_dict:
            self.h_dict[h_key] = np.zeros_like(param.data)
        h = self.h_dict[h_key]
        h += param.grad.data ** 2
        param.data -= self.lr * param.grad.data / np.sqrt(h + self.eps)


class Adam(Optimizer):
    def __init__(self, lr=0.01, beta1=0.9, beta2=0.99):
        super().__init__()
        self.lr = lr
        self.b1 = beta1
        self.b2 = beta2
        self.idx = 1
        self.momentum_dict = {}
        self.velocity_dict = {}

    def update_one(self, param):
        v_key = m_key = id(param)
        if v_key not in self.velocity_dict:
            self.velocity_dict[v_key] = np.zeros_like(param.data)
        if m_key not in self.momentum_dict:
            self.momentum_dict[m_key] = np.zeros_like(param.data)
        
        # As self.idx increases, the correction values converge to 1
        b1_correction = 1 - self.b1 ** self.idx
        b2_correction = 1 - self.b2 ** self.idx
        self.idx += 1

        m = self.momentum_dict[m_key]
        v = self.velocity_dict[v_key]

        m = self.b1 * m + (1 - self.b1) * param.grad.data
        v = self.b2 * v + (1 - self.b2) * (param.grad.data ** 2)

        m_hat = m / b1_correction
        v_hat = v / b2_correction

        param.data -= self.lr * (m_hat / np.sqrt(v_hat + 1e-8))



