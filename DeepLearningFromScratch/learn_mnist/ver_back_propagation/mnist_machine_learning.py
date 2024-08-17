import sys, os
import matplotlib.pyplot as plt
sys.path.append(os.pardir)

import numpy as np
import matplotlib.pyplot as plt
from mnist_data_preprocessing import batch_generator, l_image, l_index
from layer_net import TwoLayerNet

if len(sys.argv) > 1:
    batch_size = int(sys.argv[1])
else:
    batch_size = 60

net = TwoLayerNet(784, batch_size, 10)

idx = 1
accuracy = 0.3
x_trial = []
y_loss = []

for batch in batch_generator(l_image, l_index, batch_size):
    print(f"Batch {idx: 3}")
    y_loss.append(net.loss(batch[0], batch[1]))
    x_trial.append(idx)
    idx += 1

    net.learning(batch[0], batch[1])
    print()

x_trial = np.array(x_trial)
y_loss = np.array(y_loss)
plt.plot(x_trial, y_loss)
plt.show()
    
