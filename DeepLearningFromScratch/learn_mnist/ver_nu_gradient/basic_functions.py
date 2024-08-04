import matplotlib.pyplot as plt
import numpy as np


# Activation functions: step, sigmoid, relu
def step(x: np.ndarray) -> np.ndarray:
    tmp = x > 0
    return tmp.astype(int)

def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0, x)

def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))

# Returns a ndarray containing probabilities of each entity in the input ndarray.
def softmax(x: np.ndarray) -> np.ndarray:
    # To avoid getting crashed value by outliers.
    correction = np.max(x)
    exp_x = np.exp(x - correction)
    return exp_x / np.sum(exp_x)

# Error functions
## SSE
def sum_of_squares(y: np.ndarray, t: np.ndarray) -> float:
    return 0.5 * np.sum((y - t) ** 2)

def cross_entropy_error(y: np.ndarray, t: np.ndarray, delta=1e-7) -> float:
    return -np.sum(t * np.log(y + delta)) / t.size

# Numerical gradient
def numerical_gradient(f , x: np.ndarray) -> np.ndarray:
    h = 1e-4
    x_flat = x.ravel()
    grad = np.zeros_like(x)
    grad_flat = grad.ravel()

    if not x.flags['C_CONTIGUOUS']:
        raise ValueError('Input array must be contiguous in memory.')

    for i in range(x_flat.size):
        temp = x_flat[i]

        x_flat[i] += h
        fxh_r = f(x)

        x_flat[i] -= 2 * h
        fxh_l = f(x)

        grad_flat[i] = (fxh_r - fxh_l) / (2 * h)

        x_flat[i] = temp

    return grad


# When executed directly
if __name__ == '__main__':
    MIN = -5
    MAX = 5
    INTERVAL = 0.1
    x = np.arange(MIN, MAX, INTERVAL)
    y_step = step(x)
    y_relu = relu(x / MAX)
    y_sigmoid = sigmoid(x)

    # Plot
    plt.subplot(3, 1, 1)
    plt.plot(x, y_step, label='step')
    plt.legend()

    plt.subplot(3, 1, 2)
    plt.plot(x, y_relu, label='ReLu')
    plt.legend()

    plt.subplot(3, 1, 3)
    plt.plot(x, y_sigmoid, label='sigmoid')
    plt.legend()

    plt.show()
