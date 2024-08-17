import sys, os
import matplotlib.pyplot as plt
sys.path.append(os.pardir)

import numpy as np
import matplotlib.pyplot as plt
from mnist_data_preprocessing import batch_generator, l_image, l_index
from optimizers import *
from layer_net import TwoLayerNet

if len(sys.argv) > 1:
    batch_size = int(sys.argv[1])
else:
    batch_size = 500

net_sgd = TwoLayerNet(784, batch_size, 10, optimizer=SGD(0.001))
net_momentum = TwoLayerNet(784, batch_size, 10, optimizer=Momentum(0.001))
net_adagrads = TwoLayerNet(784, batch_size, 10, optimizer=AdaGrads(0.001))

idx = 1
x_trial_sgd = []
y_loss_sgd = []
x_trial_momentum = []
y_loss_momentum = []
x_trial_adagrads = []
y_loss_adagrads = []


for batch in batch_generator(l_image, l_index, batch_size):
    print(f"Batch {idx: 3}")
    y_loss_sgd.append(net_sgd.loss(batch[0], batch[1]))
    x_trial_sgd.append(idx)
    idx += 1

    net_sgd.learning(batch[0], batch[1])
    print()

idx = 1
for batch in batch_generator(l_image, l_index, batch_size):
    print(f"Batch {idx: 3}")
    y_loss_momentum.append(net_momentum.loss(batch[0], batch[1]))
    x_trial_momentum.append(idx)
    idx += 1

    net_momentum.learning(batch[0], batch[1])
    print()

idx = 1
for batch in batch_generator(l_image, l_index, batch_size):
    print(f"Batch {idx: 3}")
    y_loss_adagrads.append(net_adagrads.loss(batch[0], batch[1]))
    x_trial_adagrads.append(idx)
    idx += 1

    net_adagrads.learning(batch[0], batch[1])
    print()

fig, axs = plt.subplots(nrows=1, ncols=3)
axs[0].plot(x_trial_sgd, y_loss_sgd)
axs[1].plot(x_trial_momentum, y_loss_momentum)
axs[2].plot(x_trial_adagrads, y_loss_adagrads)


plt.show()
    
