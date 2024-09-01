from MyDeZero.dataset.data_classes import MNIST, DataLoader, one_hot, FastMNIST
from MyDeZero.core.common_classes import MLP, Adam, SGD, MomentumSGD
from MyDeZero.core.common_functions import *
from MyDeZero.core.core_functions import *
from MyDeZero.cuda import cuda
import matplotlib.pyplot as plt
import cupy as cp
import numpy as np

train_set = FastMNIST(train=True,\
                transform=lambda x: x.flatten(),\
                target_transform=lambda t: one_hot(t, 10))
'''
print(train_set[0][0].shape)
print(train_set[0][1])
print(len(train_set))

print(test_set[0][0].shape)
print(test_set[0][1])
print(len(test_set))
'''

train_loader = DataLoader(train_set, 100)
model1 = MLP((100, 10), activation=relu, normal=True)
model2 = MLP((100, 10), activation=relu, normal=False)
optimizer1 = Adam(0.001).setup(model1)
optimizer2 = Adam(0.001).setup(model2)

if cuda.gpu_enable:
    train_loader.to_gpu()
    model1.to_gpu()
    model2.to_gpu()
    

idx = 0
y_loss1 = []
y_loss2 = []
y_acc1 = []
y_acc2 = []
for img, label in train_loader:
    idx += 1
    pred1 = model1(img)
    pred2 = model2(img)
    loss1 = softmax_cross_entropy(pred1, label)
    loss2 = softmax_cross_entropy(pred2, label)
    acc1 = accuracy(pred1, label)
    acc2 = accuracy(pred2, label)
    y_loss1.append(loss1.data)
    y_acc1.append(acc1.data)
    y_loss2.append(loss2.data)
    y_acc2.append(acc2.data)

    model1.clear_grads()
    loss1.backward()
    optimizer1.update()

    model2.clear_grads()
    loss2.backward()
    optimizer2.update()

    if idx % 10 == 0:
        print(f'Batch {idx: 3d}')
        print(f'{loss1.data: .5f}, {acc1.data * 100: .2f}%')
        print(f'{loss2.data: .5f}, {acc2.data * 100: .2f}%')
        print()

fig, axes = plt.subplots(ncols=2)
axes[0].plot(np.arange(600), y_loss1, label='bare')
axes[0].plot(np.arange(600), y_loss2, label='normal')
axes[1].plot(np.arange(600), y_acc1, label='bare')
axes[1].plot(np.arange(600), y_acc2, label='normal')
plt.show()