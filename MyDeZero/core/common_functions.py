import numpy as np
from .core_classes import *


class Exp(Function):
    def forward(self, x: np.ndarray) -> np.ndarray:
        y = np.exp(x)

        return y

    def backward(self, grad_y):
        x = self.inputs[0]
        grad_x = exp(x) * grad_y

        return grad_x


class Log(Function):
    def forward(self, x: np.ndarray) -> np.ndarray:
        y = np.log(x)

        return y

    def backward(self, grad_y):
        x = self.inputs[0]
        grad_x = (1 / x) * grad_y

        return grad_x

class Sin(Function):
    def forward(self, x: np.ndarray) -> np.ndarray:
        y = np.sin(x)
        return y
    
    def backward(self, grad_y):
        x = self.inputs[0]
        grad_x = grad_y * cos(x)
        return grad_x

class Cos(Function):
    def forward(self, x: np.ndarray) -> np.ndarray:
        y = np.cos(x)
        return y
    
    def backward(self, grad_y):
        x = self.inputs[0]
        grad_x = grad_y * (-sin(x))
        return grad_x


class Tanh(Function):
    def forward(self, x: np.ndarray) -> np.ndarray:
        y = np.tanh(x)
        return y
    
    #(tanh(x))` = 1 - y**2
    def backward(self, grad_y: Variable) -> Variable:
        y = self.outputs[0]()
        grad_x = grad_y * (1 - y * y)
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



#-----------------------------------------------------