import numpy as np
from MyDeZero.core.common_classes import Conv2d
from MyDeZero.core.core_functions import numerical_gradient
from MyDeZero.core.common_functions import Pooling
from MyDeZero import Variable

mode = 'Conv2d'

x = Variable(np.random.randn(1, 1, 4, 4).astype(np.float32))
if mode == 'Conv2d':
    test_conv = Conv2d(3, (2, 2))
    func_x = lambda x: test_conv(x).sum()
    func_w = lambda w: test_conv(x).sum()
    func_b = lambda b: test_conv(x).sum()
    print(x.data)
    y = test_conv(x)
    z = y.sum()
    print(y.data)
    print(z)
    print(func_x(x).data)
    print(func_w(test_conv.W).data)
    print(func_b(test_conv.b).data)
    z.backward()

    nu_grad_x = numerical_gradient(func_x, x, eps=1e-4)
    nu_grad_w = numerical_gradient(func_w, test_conv.W, eps=1e-4)
    nu_grad_b = numerical_gradient(func_b, test_conv.b, eps=1e-4)
    print(x.grad)
    print(nu_grad_x)
    print()

    print(test_conv.W.grad)
    print(nu_grad_w)
    print()

    print(test_conv.b.grad)
    print(nu_grad_b)
    print()

    print((nu_grad_x/x.grad).sum()/x.grad.size)
    print((nu_grad_w/test_conv.W.grad).sum()/test_conv.W.grad.size)
    print((nu_grad_b/test_conv.b.grad).sum()/test_conv.b.grad.size)

elif mode == 'pool':
    test_pool = Pooling((2, 2), 1, 0)
    y = test_pool(x)
    y_sum = y.sum()
    func = lambda p: test_pool(p).sum()

    print(x.data)
    print(y.data)

    y_sum.backward()
    nu_grad = numerical_gradient(func, x, eps=1e-2)
    print(nu_grad[0])
    print(x.grad)
