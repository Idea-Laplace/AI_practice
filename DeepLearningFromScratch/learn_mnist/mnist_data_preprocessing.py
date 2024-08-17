from random import shuffle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import imread
import tkinter as tk
from PIL import ImageTk, Image

RESIDUE = -1

l_image_flatten = np.load('../temp_data_flattened/train_image_set.npy')
l_index_flatten = np.load('../temp_data_flattened/train_index_set.npy')
t_image_flatten = np.load('../temp_data_flattened/test_image_set.npy')
t_index_flatten = np.load('../temp_data_flattened/test_index_set.npy')

l_image = np.load('../temp_data/train_image_set.npy')
l_image = l_image.reshape(60000, RESIDUE, 28, 28)
l_index = np.load('../temp_data/train_index_set.npy')

t_image = np.load('../temp_data/test_image_set.npy')
t_image = t_image.reshape(10000, RESIDUE, 28, 28)
t_index = np.load('../temp_data/test_index_set.npy')

def batch_generator(image_set: np.ndarray, index_set: np.ndarray, size: int=10):
    assert image_set.shape[0] == index_set.shape[0],\
          'Number of images is not coordinate with that of indexs'

    IMG, IDX = 0, 1

    img_idx_pair = list(zip(image_set, index_set))
    shuffle(img_idx_pair)

    shuffled_img_set = np.array([pair[IMG] for pair in img_idx_pair])
    shuffled_idx_set = np.array([np.eye(10)[pair[IDX]] for pair in img_idx_pair])

    for i in range(0, image_set.shape[0], size):
        split = min(i + size, image_set.shape[0])
        yield shuffled_img_set[i: split], shuffled_idx_set[i: split]
    

if __name__ == '__main__':
    flatten_set = []
    flatten_set.append(l_image_flatten)
    flatten_set.append(l_index_flatten)
    flatten_set.append(t_image_flatten)
    flatten_set.append(t_index_flatten)
    
    original_set = []
    original_set.append(l_image)
    original_set.append(l_index)
    original_set.append(t_image)
    original_set.append(t_index)

    for arr in flatten_set:
        print(arr.shape)
    for arr in original_set:
        print(arr.shape)
    
