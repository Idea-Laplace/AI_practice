import numpy as np
from .core_classes import *


class Exp(Function):
    def forward(self, x: np.ndarray) -> np.ndarray:
        y = np.exp(x)

        return y

    def backward(self, grad_y: np.ndarray) -> np.ndarray:
        x = self.inputs[0].data
        grad_x = np.exp(x) * grad_y

        return grad_x


class Log(Function):
    def forward(self, x: np.ndarray) -> np.ndarray:
        y = np.log(x)

        return y

    def backward(self, grad_y: np.ndarray) -> np.ndarray:
        x = self.inputs[0].data
        grad_x = (1 / x) * grad_y

        return grad_x

#----------------------------------------------------
def exp(x: Variable) -> Variable:
    return Exp()(x)

def log(x: Variable) -> Variable:
    return Log()(x)

#-----------------------------------------------------