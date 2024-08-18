from MyDeZero import Variable
from MyDeZero import numerical_gradient
from MyDeZero import get_dot_graph, plot_dot_graph
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


x = Variable(1.0, 'x')
y = Variable(1.0, 'y')
z = sphere(x, y)
z.backward()

x.clear_grad(), y.clear_grad()
print((x.grad, y.grad))


z = matyas(x, y)
z.backward()
print(x.grad, y.grad)

x.clear_grad(), y.clear_grad()
z = Goldstein_Price(x, y)
z.name = 'z'
z.backward()
print(z.data)
print((x.grad, y.grad))

#gold_graph = get_dot_graph(z, verbose=True)
#plot_dot_graph(z, 'goldstein.png')

nu_grads = numerical_gradient(Goldstein_Price, x, y)
print(nu_grads)

x0 = Variable(0.0, 'x0')
x1 = Variable(1.0, 'x1') 
x2 = Variable(2.0, 'x2')
x3 = Variable(3.0, 'x3')
x4 = Variable(4.0, 'x4')
z = normalization(x0, x1, x2, x3, x4)
print(z)
plot_dot_graph(z[0], verbose=True, to_file='normalization.png')