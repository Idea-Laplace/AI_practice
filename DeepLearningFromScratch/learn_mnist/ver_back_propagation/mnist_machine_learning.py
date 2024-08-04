import sys, os
sys.path.append(os.pardir)

import numpy as np
import matplotlib.pyplot as plt
from mnist_data_preprocessing import batch_generator, l_image, l_index
from layer_net import TwoLayerNet

if len(sys.argv) > 1:
    batch_size = int(sys.argv[1])
else:
    batch_size = 30

net = TwoLayerNet(784, batch_size, 10)

idx = 1
accuracy = 0.3
learning_idx = []
for batch in batch_generator(l_image, l_index, batch_size):
    print(f"Batch {idx: 3}")

    if net.accuracy(batch[0], batch[1]) > accuracy:
        accuracy += 0.1
        learning_idx.append(idx)

    idx += 1

    net.learning(batch[0], batch[1])
    print()

    

print(learning_idx)