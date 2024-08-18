import sys
sys.path.append('..')

from MyFramework import Variable
from MyFramework import numerical_gradient
import numpy as np

def sphere(x: Variable, y: Variable):
    z = x**2 + y**2
    return z

def matyas(x: Variable, y: Variable):
    z = 0.26 * (x**2 + y**2) - 0.48 * x * y
    return z

def Goldstein_Price(x: Variable, y: Variable):
    z = (1 + (x+y+1)**2 * (19 - 14*x + 3*x**2 - 14*y + 6*x*y + 3*y**2)) *\
        (30 + (2*x - 3*y)**2 * (18 - 32*x + 12*x**2 + 48*y - 36*x*y + 27*y**2)) 
    return z

x = Variable(1.0)
y = Variable(1.0)
z = sphere(x, y)
z.backward()

x.clear_grad(), y.clear_grad()
print(x.grad, y.grad)


z = matyas(x, y)
z.backward()
print(x.grad, y.grad)

x.clear_grad(), y.clear_grad()
z = Goldstein_Price(x, y)
z.backward()
print(z.data)
print(x.grad, y.grad)

nu_grads = numerical_gradient(Goldstein_Price, x, y)
print(nu_grads)