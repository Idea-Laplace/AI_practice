from MyDeZero.dataset.data_classes import DataLoader, Spiral, get_spiral
from MyDeZero.core.common_classes import MLP, Adam, SGD, MomentumSGD
from MyDeZero.core.common_functions import *
from MyDeZero.core.core_functions import *
import matplotlib.pyplot as plt
import numpy as np


'''
x, t = get_spiral(shuffle=False)
plt.scatter(x[:100, 0], x[:100, 1], marker='o', color='y', label='class0')
plt.scatter(x[101:200, 0], x[101:200, 1], marker='x', color='k', label='class1')
plt.scatter(x[201:300, 0], x[201:300, 1], marker='^', color='g', label='class2')
plt.legend(loc='best')
#plt.show()
'''

#----------------------------------------------------------------
max_epoch = 300
batch_size = 30
hidden_size = 10
lr = 0.03

train_set = Spiral(train=True)
test_set = Spiral(train=False)
spiral_loader = DataLoader(train_set, batch_size, True)
model = MLP((hidden_size, 3))
optimizer = Adam(lr).setup(model)


loss_y = []
for epoch in range(max_epoch):
    loss_per_epoch = 0
    for data, label in spiral_loader:
        pred = model(data)

        loss = softmax_cross_entropy(pred, label)
        acc = accuracy(pred, label)
        model.clear_grads()
        loss.backward()
        optimizer.update()
        loss_per_epoch += loss.data


    loss_per_epoch /= (train_set.data.shape[0] / data.shape[0])
    loss_y.append(loss_per_epoch)

plt.plot(np.arange(max_epoch), np.array(loss_y))
plt.show()




