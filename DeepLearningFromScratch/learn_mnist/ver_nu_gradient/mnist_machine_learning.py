import sys, os
sys.path.append(os.pardir)

import numpy as np
import matplotlib.pyplot as plt
from mnist_data_preprocessing import batch_generator, l_image_flatten, l_index_flatten
from layer_net import TwoLayerNet

batch_size = int(sys.argv[1])

net = TwoLayerNet(784, 20, 10)

for batch in batch_generator(l_image_flatten, l_index_flatten, batch_size):
    net.learning(batch[0], batch[1])
