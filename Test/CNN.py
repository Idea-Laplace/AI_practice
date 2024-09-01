from MyDeZero import Variable
from MyDeZero.dataset.data_classes import FastMNIST, DataLoader, one_hot, MNIST
from MyDeZero.core.common_classes import CNNMLP, CNN, MLP, Adam, SGD
import MyDeZero.core.core_functions as crf
import MyDeZero.core.common_functions as cmf
import matplotlib.pyplot as plt
from MyDeZero.cuda import cuda
import numpy as np
import cupy as cp
import os

max_epoch = 5
BATCH = 30
DROPOUT = 0.3
lr = 1e-3
mnist = FastMNIST(train=True, target_transform=lambda t: one_hot(t, 10))
train_loader = DataLoader(mnist, BATCH)
mlp = MLP((100, 10), activation=cmf.relu, normal=True, dropout=DROPOUT)
cnn = CNN((16, 16, 32, 32), (2, 2), pooling=True, activation=cmf.relu)
model = CNNMLP((cnn,), mlp)
optimizer = Adam(lr).setup(model)

if os.path.exists('my_cnn.npz'):
    model.load_weights('my_cnn.npz')

if cuda.gpu_enable:
    train_loader.to_gpu()
    model.to_gpu()

batch = 1
y_loss = []
y_acc = []



for epoch in range(max_epoch):
    print(f'\nepoch {epoch + 1}')
    for img, idx in train_loader:
        N, C, H, W = img.shape

        pred = model.forward(img)
        loss = cmf.softmax_cross_entropy(pred, idx)
        acc = cmf.accuracy(pred, idx)

        y_loss.append(loss.data)
        y_acc.append(acc.data)
        if batch % 100 == 0 or batch ==1:
            print()
            print('  Batch      Loss     Accuracy')
            print('------------------------------')
        if batch % 10 == 0 or batch == 1:
            print(f'Batch {batch: 3d}: {loss.data: .5f}   {100 * acc.data: .2f}%')
            print(cmf.softmax(pred).data.max(axis=1))
            print(pred.data.argmax(axis=1))
            print(idx.argmax(axis=1))
        model.clear_grads()
        loss.backward()

        optimizer.update()
        batch += 1

model.save_weights('my_cnn.npz')



fig, axes = plt.subplots(ncols=2)
x = np.arange(batch - 1)
y_loss = cp.array(y_loss)
y_loss = y_loss.get()
y_acc = cp.array(y_acc)
y_acc = y_acc.get()
print(y_loss.sum() / y_loss.size)
print(y_acc.sum() / y_acc.size)
axes[0].plot(x, y_loss)
axes[1].plot(x, y_acc)
plt.show()
