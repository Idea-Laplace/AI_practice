from MyDeZero import Variable
from MyDeZero.dataset.data_classes import FastMNIST, DataLoader, one_hot, MNIST
from MyDeZero.core.common_classes import CNNMLP, CNN, MLP, Adam, SGD
import MyDeZero.core.core_functions as crf
import MyDeZero.core.common_functions as cmf
import matplotlib.pyplot as plt
import numpy as np

BATCH = 100
DROPOUT = None
mnist = FastMNIST(train=True, target_transform=lambda t: one_hot(t, 10))
train_loader = DataLoader(mnist, BATCH)
mlp = MLP((100, 10), activation=cmf.relu, normal=True, dropout=DROPOUT)
cnn = CNN((16, 16, 32, 32), (2, 2), pooling=True, activation=cmf.relu)
model = CNNMLP((cnn,), mlp)
optimizer1 = Adam(0.001).setup(cnn)
optimizer2 = Adam(0.001).setup(mlp)
optimizer3 = Adam(0.001).setup(model)

batch = 1
mode = 3
y_loss = []
y_acc = []

for img, idx in train_loader:
    N, C, H, W = img.shape
    if mode == 1:
        pred = cnn.forward(img)
        pred = pred.reshape(N, -1)
        pred = mlp.forward(pred)
        loss = cmf.softmax_cross_entropy(pred, idx)
        acc = cmf.accuracy(pred, idx).data
        '''
        if batch % 10 == 0:
            print(f'Batch {batch: 3d}: {loss.data: .5f}, {acc: .2f}')
        '''

        mlp.clear_grads()
        cnn.clear_grads()
        loss.backward()
        optimizer1.update()
        optimizer2.update()

        batch += 1

    elif mode == 2:
        pred = mlp.forward(img.reshape(N, -1))
        loss = cmf.softmax_cross_entropy(pred, idx)
        acc = cmf.accuracy(pred, idx).data
        if batch % 10 == 0:
            print(f'Batch {batch: 3d}: {loss.data: .5f}, {acc: .2f}')
        mlp.clear_grads()
        loss.backward()
        optimizer2.update()
        batch +=1
    
    elif mode == 3:
        pred = model.forward(img)
        loss = cmf.softmax_cross_entropy(pred, idx)
        acc = cmf.accuracy(pred, idx)
        y_loss.append(loss.data)
        y_acc.append(acc.data)
        if batch % 100 == 0 or batch ==1:
            print('  Batch      Loss     Accuracy')
            print('------------------------------')
        if batch % 10 == 0 or batch == 1:
            print(f'Batch {batch: 3d}: {loss.data: .5f}, {100 * acc.data: .2f}%')
        model.clear_grads()
        loss.backward()
        optimizer3.update()
        batch += 1

fig, axes = plt.subplots(ncols=2)
x = np.arange(len(mnist) / BATCH)
y_loss = np.array(y_loss)
y_acc = np.array(y_acc)
print(y_loss.sum() / y_loss.size)
print(y_acc.sum() / y_acc.size)
axes[0].plot(x, y_loss)
axes[1].plot(x, y_acc)
plt.show()