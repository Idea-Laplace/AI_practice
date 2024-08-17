import numpy as np
import sys, os
sys.path.append(os.pardir)
from Variables.variable import Variable


# The class 'Function' is a super class
# for specific function subclasses.

class Function:
    def __call__(self, *inputs: Variable) -> Variable:
        # Extracting data to apply to the method 'forward'
        xs = [x.data for x in inputs]
        ys = self.forward(*xs)
        # The generation of a 'Function' is determined by its input generation
        self.generation = max([input.generation for input in inputs])

        # To be consistent with tuple type, which is iterable
        if not isinstance(ys, tuple):
            ys = (ys,)

        # A new variable instance
        # When a subroutine returns multiple variables
        outputs = [Variable(as_array(y)) for y in ys] # Refer to as_array function in the Supportive function section.

        for output in outputs:
            output.set_creator(self)

        # Stores the recent call information
        self.inputs = inputs
        self.outputs = outputs

        # tuple of Variables or a single Variable
        return outputs if len(outputs) > 1 else outputs[0]

    def forward(self, *input: np.ndarray) -> np.ndarray:
        raise NotImplementedError()

    def backward(self, output_grad: np.ndarray) -> np.ndarray:
        raise NotImplementedError()


# Subclass --------------------------------------------------------

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


class Add(Function):
    def forward(self, *xs: np.ndarray) -> np.ndarray:
        y = sum(xs)

        return y

    def backward(self, grad_y: np.ndarray) -> np.ndarray:
        grad_xs = (grad_y,) * len(self.inputs)

        return grad_xs


class Multiply(Function):
    def forward(self, *xs: np.ndarray) -> np.ndarray:
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


# Supportive functions --------------------------------------------

'''
When a numpy array of zero-dimension go through a numpy method,
there would be a chance that the the type of output instance is
just a scalar type, not a numpy array, which is desired.
To avoid this, the following as_array function checks whether the
output is scalar or not, and converts it to a numpy array if necessary.
'''
def as_array(x) -> np.ndarray:
    if np.isscalar(x):
        return np.array(x)
    else:
        return x

def exp(x: Variable)-> Variable:
    return Exp()(x)

def log(x: Variable) -> Variable:
    return Log()(x)

def add(*xs: Variable) -> Variable:
    return Add()(*xs)

def mul(*xs: Variable) -> Variable:
    return Multiply()(*xs)

if __name__ == '__main__':
    a = np.array(2)
    b = np.array(3)
    c = np.array(4)
    d = np.array(0)

    A = Variable(a)
    B = Variable(b)
    C = Variable(c)
    D = Variable(d)
    E = add(A, B, C, D)
    E.backward()

    print(E.data)
    print(D.grad)
    print(C.grad)
    print(B.grad)
    print(A.grad)

    A.clear_grad()
    B.clear_grad()
    C.clear_grad()
    D.clear_grad()

    print()
    E = mul(A, B, C)
    E.backward()

    print(E.data)
    print(C.grad)
    print(B.grad)
    print(A.grad)

    A.clear_grad()
    B.clear_grad()
    C.clear_grad()
    D.clear_grad()

    print()
    E = mul(A, B, C, D)
    E.backward()

    print(E.data)
    print(D.grad)
    print(C.grad)
    print(B.grad)
    print(A.grad)

