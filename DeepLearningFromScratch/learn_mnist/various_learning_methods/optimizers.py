import numpy as np

# Super class definition
class Optimizer:
    def __init__(self, lr):
        self.lr = lr
    def update(self, params, grade):
        raise NotImplementedError("Subclass must implement abstract method")
    

# Child class definitions
class SGD(Optimizer):
    def update(self, params, grads):
        for key in params.keys():
            params[key] -= self.lr * grads[key]

class Momentum(Optimizer):
    def __init__(self, lr, momentum = 0.2):
        super().__init__(lr)
        self.momentum = momentum
        self.v = None

    def update(self, params, grads):
        if self.v is None:
            self.v = {}
            for key, val in params.items():
                self.v[key] = np.zeros_like(val)
        
        for key in params:
            self.v[key] = self.momentum * self.v[key] - self.lr * grads[key]
            params[key] += self.v[key]


class AdaGrads(Optimizer):
    def __init__(self, lr, div_zero = 1e-7):
        super().__init__(lr)
        self.h = None
        self.div_zero = div_zero

    def update(self, params, grads):
        if self.h is None:
            self.h = {}
            for key, val in params.items():
                self.h[key] = np.zeros_like(val)
        
        for key in params:
            self.h[key] += grads[key] * grads[key]
            params[key] -= self.lr * grads[key] / (np.sqrt(self.h[key]) + self.div_zero)


