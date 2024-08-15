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
    def __init__(self, lr, momentum = 0.9):
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


class Adam(Optimizer):
    def __init__(self, lr=0.001, b1=0.9, b2=0.99):
        super().__init__(lr)
        self.b1 = b1
        self.b2 = b2
        self.idx = 1
        self.m = None
        self.v = None

    def update(self, params, grads):
        if self.m is None:
            self.m = {}
            for key, val in params.items():
                self.m[key] = np.zeros_like(val)
                
        if self.v is None:
            self.v = {}
            for key, val in params.items():
                self.v[key] = np.zeros_like(val)
        
        bias_b1 = 1 - (self.b1 ** self.idx)
        bias_b2 = 1 - (self.b2 ** self.idx)
        self.idx += 1

        for key in params:
            self.m[key] = self.b1 * self.m[key] + (1 - self.b1) * grads[key]
            self.v[key] = self.b2 * self.v[key] + (1 - self.b2) * (grads[key] * grads[key])
            m_hat = self.m[key] / bias_b1
            v_hat = self.v[key] / bias_b2

            params[key] -= self.lr * (m_hat / (np.sqrt(v_hat) + 1e-8))