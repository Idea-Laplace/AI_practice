import unittest
import sys, os
sys.path.append(os.pardir)
import numpy as np
from Variables.variable import *
from Functions.function import *


# ~/AI_practice/MyFramework/Variables/variable.py

class VariableTest(unittest.TestCase):
    def test_init(self):
        sample = np.array(4.0)
        x = Variable(sample)
        self.assertEqual(x.data, sample)
        self.assertEqual(x.grad, None)

    def test_set_creator(self):
        sample = np.array(3.0)
        x = Variable(sample)
        y = exp(x)
        self.assertEqual(type(y.creator), Exp)
        self.assertEqual(y.data, np.exp(x.data))


# ~/AI_practice/MyFramework/Functions/function.py

def numerical_gradient(f: Function, x: Variable):
    eps = 1e-5
    # Use ravel(), not flatten(), as ravel() indicates the same instance but view it as flat.
    # Even if the x is of ndim zero, the ravel automatically cast x_flatten into ndim 1
    x_flatten = x.data.ravel()
    grad_x = np.zeros_like(x_flatten)

    # The algorithm of this 'numerical_gradient' refers to the 'centered divided difference' method.
    # ,in which the formula is represented by f'(x) = lim_(h->0)[(f(x+h) - f(x-h)) / (2*h)]
    for i in range(x_flatten.size):
        x_flatten[i] += eps
        fxr = f(x)
        
        x_flatten[i] -= 2 * eps
        fxl = f(x)

        grad_x[i] = (fxr.data - fxl.data) / (2 * eps) 
    
    grad_x = grad_x.reshape(x.data.shape)

    return grad_x


class ExpTest(unittest.TestCase):
    def test_forward(self):
        sample = np.array(3.0)
        x = Variable(sample)
        y = exp(x)

        self.assertEqual(y.data, np.exp(3.0))

    def test_backward(self):
        sample = np.array(3.0)
        x = Variable(sample)
        y = exp(x)
        y.backward()

        expected = np.exp(np.array(3.0))
        self.assertEqual(x.grad, expected)
        temp = lambda dummy: y.creator(x)
        self.assertTrue(np.allclose(x.grad, numerical_gradient(temp, x)))


class LogTest(unittest.TestCase):
    def test_forward(self):
        sample = np.array(3.0)
        x = Variable(sample)
        y = log(x)

        self.assertEqual(y.data, np.log(3.0))

    def test_backward(self):
        sample = np.array(3.0)
        x = Variable(sample)
        y = log(x)
        y.backward()

        expected = 1 / np.array(3.0)
        self.assertEqual(x.grad, expected)
        temp = lambda dummy: y.creator(x)
        self.assertTrue(np.allclose(x.grad, numerical_gradient(temp, x)))


class AddTest(unittest.TestCase):
    def test_forward(self):
        sample0, sample1 = np.array(3), np.array(2)
        x0 = Variable(sample0)
        x1 = Variable(sample1)
        y = add(x0, x1)

        self.assertEqual(y.data, np.array(5))

    def test_backward(self):
        sample0, sample1 = np.array(3.), np.array(2.)
        x0 = Variable(sample0)
        x1 = Variable(sample1)
        y = add(x0, x1)
        y.backward()

        self.assertEqual(x0.grad, y.grad)
        self.assertEqual(x1.grad, y.grad)
        temp = lambda dummy: y.creator(x0, x1)
        self.assertTrue(np.allclose(x0.grad, numerical_gradient(temp, x0)))
        self.assertTrue(np.allclose(x1.grad, numerical_gradient(temp, x1)))


class MultiplyTest(unittest.TestCase):
    def test_forward(self):
        sample0, sample1 = np.array(3.0), np.array(2.0)
        x0 = Variable(sample0)
        x1 = Variable(sample1)
        y = mul(x0, x1)

        self.assertEqual(y.data, np.array(3.0 * 2.0))

    def test_backward(self):
        sample0, sample1 = np.array(3.0), np.array(2.0)
        x0 = Variable(sample0)
        x1 = Variable(sample1)
        y = mul(x0, x1)
        y.backward()

        self.assertEqual(x0.grad, np.array(x1.data * y.grad))
        self.assertEqual(x1.grad, np.array(x0.data * y.grad))
        temp = lambda dummy: y.creator(x0, x1)
        self.assertTrue(np.allclose(x0.grad, numerical_gradient(temp, x0)))
        self.assertTrue(np.allclose(x1.grad, numerical_gradient(temp, x1)))

# ----------------------------------------------------------------------
class ScalarNetworkTest(unittest.TestCase):
    def test_composite_function(self):
        coefficient_exp = np.array(2.0)
        coefficient_bias = np.array(9.0)
        coefficient_0 = np.array(-1.0)
        coefficient_1 = np.array(3.0)
        coefficient_2 = np.array(-3.0)
        coefficient_3 = np.array(1.0)
        variable_x = np.array(2.0)

        A = Variable(coefficient_exp)
        B = Variable(coefficient_bias)
        d = Variable(coefficient_0)
        c = Variable(coefficient_1)
        b = Variable(coefficient_2)
        a = Variable(coefficient_3)
        x = Variable(variable_x)

        pol_term0 = d
        pol_term1 = mul(c, x)
        pol_term2 = mul(b, x, x)
        pol_term3 = mul(a, x, x, x)
        polynomial = add(pol_term0, pol_term1, pol_term2, pol_term3)

        # y = 5*exp((x-1)^3) + 9
        # y' = 15((x-1)^2)exp((x-1)^3)
        
        y = add(mul(A, exp(polynomial)), B)
        y.backward()

        def dummy(A, B, a, b, c, d, x):
            pol_term0 = d
            pol_term1 = mul(c, x)
            pol_term2 = mul(b, x, x)
            pol_term3 = mul(a, x, x, x)
            polynomial = add(pol_term0, pol_term1, pol_term2, pol_term3)
            y = add(mul(A, exp(polynomial)), B)

            return y


        vars = [polynomial, pol_term0, pol_term1, pol_term2, pol_term3, a, b, c, d, x]
        print(y.data)
        print(y.creator)
        dump = lambda x: dummy(A, B, a, b, c, d, x)
        for var in vars:
            print(var.grad)
        self.assertTrue(np.allclose(x.grad, numerical_gradient(dump, x)))



if __name__ == '__main__':
    unittest.main()


