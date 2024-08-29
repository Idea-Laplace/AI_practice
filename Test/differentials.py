from MyDeZero import Variable, as_variable
from MyDeZero.core.common_functions import *
from MyDeZero.core.core_functions import *
from MyDeZero import numerical_gradient
from MyDeZero import get_dot_graph, plot_dot_graph
import matplotlib.pyplot as plt

import numpy as np
import math

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

def arithmetic_mean(*xs: Variable) -> Variable:
    output = sum(xs) / len(xs)
    output.name = 'mean'
    return output

def variance(*xs: Variable) -> Variable:
    mean = arithmetic_mean(*xs)
    output = sum([(x-mean)**2 for x in xs]) / len(xs)
    output.name = 'variance'
    return output 

def normalization(*xs: Variable) -> tuple[Variable]:
    mean = arithmetic_mean(*xs)
    std_dev = variance(*xs) ** 0.5
    std_dev.name = 'standard deviation'
    output = tuple([(x-mean)/std_dev for x in xs])
    for i, norm_x in enumerate(output):
        norm_x.name = f'normalized_x{i}'
    return output

def taylor_sin(x: Variable, threshold=1e-150):
    y = 0
    i = 0
    while True:
        c = (-1)**i / math.factorial(2*i + 1)
        t = c * x**(2*i + 1)
        y = y + t
        i += 1
    
        if abs(t.data) < threshold:
            break
    return y

def rosenbrock(x0: Variable, x1: Variable, coeff: tuple[int]=(100, 1)) -> Variable:
    y = coeff[0]*(x1 - x0**2)**2 + coeff[1]*(1 - x0)**2
    return y

def temp_subroutine(newton_method=True):
    x0 = Variable(0.0)
    x1 = Variable(2.0)
    lr = 0.001
    iters = 10

    x0s, x1s = [], []

    for i in range(iters):

        y = rosenbrock(x0, x1)
        x0s.append(as_variable(x0.data.copy()))
        x1s.append(as_variable(x1.data.copy()))
        x0.clear_grad()
        x1.clear_grad()

        y.backward()
        if newton_method:
            gx0 = x0.grad
            gx1 = x1.grad
            x0.clear_grad()
            x1.clear_grad()
            gx0.backward()
            gx1.backward()
            gx0_2 = x0.grad
            gx1_2 = x1.grad
            x0.data -= gx0.data / gx0_2.data
            x1.data -= gx1.data / gx1_2.data
        else:
            x0.data -= lr * x0.grad.data
            x1.data -= lr * x1.grad.data
        

    return x0s, x1s
    
'''
x = np.arange(-1, 1, 0.01)
y = np.arange(-1, 2, 0.01)
x_m, y_m = grid = np.meshgrid(x, y)
z = 100 * (y_m - x_m**2)**2 + (1 - x_m)**2
fig = plt.figure()
g1 = fig.add_subplot(111, projection='3d')
g1.plot_surface(x_m, y_m, z, color='lightblue')

x0s = []
x1s = []
x0s, x1s = temp_subroutine()
rosen = []
for x0, x1 in zip(x0s, x1s):
    rosen.append(rosenbrock(x0, x1) +1)
x0s = [x0.data for x0 in x0s]
x1s = [x1.data for x1 in x1s]
rosen = [y.data for y in rosen]
g1.plot(x0s, x1s, rosen, color='r', linewidth=5)

plt.show()
'''

'''
x = Variable(np.linspace(-7, 7, 200))
y = sin(x)
y.backward(create_graph=True)

logs = [y.data]

for i in range(3):
    logs.append(x.grad.data)
    gx = x.grad
    x.clear_grad()
    gx.backward(create_graph=True)
    print(i)

labels =  ['y=sin(x)', 'y`', 'y``', 'y```']
for i, v in enumerate(logs):
    plt.plot(x.data, logs[i], label=labels[i])
plt.legend(loc='lower right')
plt.show()
'''
'''
x = Variable(1.0)
y = tanh(x)
x.name = 'x'
y.name = 'y'
y.backward(create_graph=True)

iters = 6

for i in range(iters):
    grad_x = x.grad
    x.clear_grad()
    grad_x.backward(create_graph=True)

grad_x = x.grad
grad_x.name = 'grad_x' + str(iters+1)
plot_dot_graph(grad_x, verbose=False, to_file='tanh.png')
'''

'''
a = Variable(np.array([1, 2, 3, 4]))
b = a.reshape(2, 2)
print(a)
print(b)

a = Variable(np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]))
b = a.transpose(2, 0, 1)
print(b)
c = Transpose((2, 0, 1))
d = c.backward(b)
print(d)

e = a.sum(axis=0, keepdims=True)
print(e)
'''

'''
a = Variable(np.array([1, 2, 3]))
b = broadcast_to(a, (3, 3, 3))
b.backward()
print(b)
c = sum_to(b, (1,3))
print(c)
'''

w = Variable(np.array([[1, 2, 3], [4, 5, 6]]))
x = Variable(np.array([2, 1]))
z = mat_mul(x, w)
print(z)
z.backward()
print(x.grad)
print(w.grad)
