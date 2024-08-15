import sys, os
import matplotlib.pyplot as plt
sys.path.append(os.pardir)

import numpy as np
import matplotlib.pyplot as plt
from mnist_data_preprocessing import *
from optimizers import *
from Conv_net import CNN

if len(sys.argv) > 1:
    batch_size = int(sys.argv[1])
else:
    batch_size = 30


CNN_test = CNN((batch_size, 1, 28 ,28))

idx = 1
x_trial = []
y_loss = []
y_accuracy = []


for batch in batch_generator(l_image, l_index, batch_size):
    print(f"Batch {idx: 3}")
    y_loss.append(CNN_test.loss(batch[0], batch[1]))
    y_accuracy.append(CNN_test.accuracy(batch[0], batch[1]))
    x_trial.append(idx)

    idx += 1

    CNN_test.learning(batch[0], batch[1])
    print()


plt.plot(x_trial, y_loss)
plt.show()
    
