from MyDeZero.core.common_functions import *
from MyDeZero.core.core_functions import *
from MyDeZero.core.common_classes import *
import numpy as np
from MyDeZero import Variable


x = Variable(np.array([[10.], [20.], [30.], [40.], [50.]]))
model = MLP((100, 1))
y = (model(x)**6).sum()
print(y)
y.backward()
func = lambda x: (model(x)**6).sum()
print(func(x))
nu_grad = numerical_gradient(func, x)
print(x.grad)
print(nu_grad)
