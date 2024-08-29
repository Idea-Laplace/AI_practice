import numpy as np
import matplotlib.pyplot as plt
from MyDeZero import Variable, Linear, Layer
import MyDeZero.core.common_functions as cmf
import MyDeZero.core.core_functions as crf
from MyDeZero.core.common_classes import MomentumSGD, SGD, AdaGrads, MLP, Adam

np.random.seed(0)
LEARNING_RATE = 0.2
ITERS = 10000

x = Variable(np.random.rand(100, 1))
x.data.sort(axis=0)
y = Variable(np.sin(2 * np.pi * x.data) + np.random.rand(100, 1))

model = MLP((10, 1), activation=cmf.sigmoid)
#optimizer = SGD()
#optimizer = MomentumSGD()
#optimizer = AdaGrads(0.2)
optimizer = Adam(0.001)
optimizer.setup(model)


for i in range(ITERS):
    y_pred = model.forward(x)
    loss = cmf.mse(y_pred, y)

    model.clear_grads()
    loss.backward()

    optimizer.update()
    
    if i % 2000 == 0:
        plt.plot(x.data, y_pred.data, label=f'Learn {i: 4d}')
        print(loss)

plt.plot(x.data, y_pred.data, label=f'Learn 9999')
plt.legend(loc='best')
print(loss)

plt.scatter(x.data, y.data)
plt.xlim(0, 1)
plt.ylim(-1, 2)
plt.show()