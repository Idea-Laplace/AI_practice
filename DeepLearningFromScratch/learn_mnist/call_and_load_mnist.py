from tensorflow.keras.datasets import mnist
import numpy as np
import os
from matplotlib import pyplot as plt


(train_image_set, train_index_set), (test_image_set, test_index_set) = mnist.load_data()

loop = True
while loop:
    arg = input("Do you want to flatten ndarray into 1?[y/n] > ")
    if arg not in {'y', 'Y', 'n', 'N'}:
        print("Enter y[Y] or n[N].")
        continue
    elif arg == 'y' or arg == 'Y':
        train_image_set = train_image_set.reshape(60000, 784)
        test_image_set = test_image_set.reshape(10000, 784)

        temp_dir = './temp_data_flattened'
        os.makedirs(temp_dir, exist_ok=True)
        loop = False
    else:
        temp_dir = './temp_data'
        os.makedirs(temp_dir, exist_ok=True)
        loop = False



np.save(temp_dir + "/train_image_set", train_image_set)
np.save(temp_dir + "/train_index_set", train_index_set)
np.save(temp_dir + "/test_image_set", test_image_set)
np.save(temp_dir + "/test_index_set", test_index_set)

if __name__ == '__main__':
    print(f"train_image_set: type {type(train_image_set)}")
    print(f"train_image_set: dtype {train_image_set.dtype}")
    print(f"train_index_set: type {type(train_index_set)}")
    print(f"train_index_set: dtype {train_index_set.dtype}")

    print(f"train_image_set: {train_image_set.shape}")
    print(f"train_index_set: {train_index_set.shape}")
    print(f"test_image_set: {test_image_set.shape}")
    print(f"test_index_set: {test_index_set.shape}")
