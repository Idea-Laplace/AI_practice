import numpy as np
import matplotlib.pyplot as plt
from MyDeZero import Variable
import MyDeZero.core.core_functions as crf
import MyDeZero.core.common_functions as cmf

# Fix seed for reproducibility
np.random.seed(0)

x = np.random.rand(100, 1) # 100 rows 1 column with a half-closed interval [0.0, 1.0)
y = 5 + 2 * x + np.random.rand(100, 1) # Ditto.

x = Variable(x)
y = Variable(y)

W = Variable(np.zeros((1, 1)))
b = Variable(5 * np.ones(1))

plt.scatter(x.data, y.data)
plt.title('Linear Regression')
plt.xlabel('x_axis')
plt.ylabel('y_axis')
#----------------------------------------------------


iters = 1000
lr = 0.1

for i in range(iters):
    y_pred = crf.linear_transform(x, W, b)
    loss = cmf.mse(y, y_pred)

    W.clear_grad()
    b.clear_grad()
    loss.backward()

    W.data -= lr * W.grad.data
    b.data -= lr * b.grad.data
    print(W, b, loss)

    if i % 50 == 0:
        plt.plot(x.data, y_pred.data)

# scatter graph
plt.show()