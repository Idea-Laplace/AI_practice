import numpy as np
from MyDeZero.cuda import cuda
import math
import random
import os

DATA = 0
LABEL = 1
PATH = os.path.dirname(__file__)

class Dataset:
    def __init__(self, train: bool=True,\
                transform: callable=None,\
                target_transform: callable=None):
        self.train = train
        self.transform = transform
        self.target_transform = target_transform

        if self.transform is None:
            self.transform = lambda x: x
        if self.target_transform is None:
            self.target_transform = lambda x: x
        
        self.data = None
        self.label = None
        self.prepare()
    
    def __getitem__(self, index: int):
        assert np.isscalar(index)
        if self.label is None:
            # self.data is of type np.ndarray,
            # hence self.data[None] is a copy of self.data.
            return self.transform(self.data[index]), None
        else:
            return self.transform(self.data[index]),\
                 self.target_transform(self.label[index])
        
    def __len__(self):
        return len(self.data)
    
    def prepare(self):
        # Here is where self.data and self.label are assigned.
        raise NotImplementedError()



class BigData(Dataset):
    def __init__(self, train: bool=True,\
                transform: callable=None,\
                target_transform: callable=None):
        super().__init__(train, transform, target_transform)
        self._path = os.path.dirname(__file__)
        self._len = None

    # Overriding __getitem__
    def __getitem__(index):
        # This is just an example, you'd need to override
        x = np.load('data/{index}.npy')
        t = np.load('label/{index}.npy')
        return x, t
    
    # Override __len__
    def __len__():
        pass

    def prepare(self):
        pass
    

#-----------------------------------------------------------------------------------
class DataLoader:
    def __init__(self, dataset, batch_size: int, shuffle=True, gpu=False):
        # dataset should have definition for __getitem__ and __len__ magic methods
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.data_size = len(dataset)
        self.max_iter = math.ceil(self.data_size / batch_size)
        self.gpu = gpu
        self.reset()
    
    def reset(self):
        self.iteration = 0
        if self.shuffle:
            # Randomly shuffles [0, 1, ..., len - 1]
            self.index = np.random.permutation(len(self.dataset))
        else:
            self.index = np.arange((len(self.dataset)))
    
    def __iter__(self):
        return self
    
    def __next__(self):
        if self.iteration >= self.max_iter:
            self.reset()
            raise StopIteration
        
        i, batch_size = self.iteration, self.batch_size
        batch_index = self.index[i * batch_size : (i + 1) * batch_size]
        batch = [self.dataset[i] for i in batch_index]
        xp = cuda.cupy if self.gpu else np
        x = xp.array([example[DATA] for example in batch])
        t = xp.array([example[LABEL] for example in batch])

        self.iteration += 1
        return x, t
    
    def next(self):
        return self.__next__()
    
    def to_cpu(self):
        self.gpu = False

    def to_gpu(self):
        self.gpu = True

#-------------------------------------------------------------------------
class Spiral(Dataset):
    def prepare(self):
        self.data, self.label = get_spiral(self.train)


class MNIST(BigData):
    def __getitem__(self, index: int):

        if self.train:
            x = np.load(self._path + '/mnist/train_set.npz')['image_set'][index]
            t = np.load(self._path + '/mnist/train_set.npz')['label_set'][index]
        else:
            x = np.load(self._path + '/mnist/test_set.npz')['image_set'][index]
            t = np.load(self._path + '/mnist/test_set.npz')['label_set'][index]
        return self.transform(x), self.target_transform(t)
    
    def __len__(self):
        if self._len is None:
            if self.train:
                self._len = len(np.load(self._path + '/mnist/train_set.npz')['label_set']) 
            else:
                self._len = len(np.load(self._path + '/mnist/test_set.npz')['label_set'])
        return self._len


class FastMNIST(Dataset):
    def prepare(self):
        if self.train:
            self.data, self.label = np.load(PATH + '/mnist/train_set.npz').values()
        else:
            self.data, self.label = np.load(PATH + '/mnist/test_set.npz').values()



def get_spiral(train=True, shuffle=True):
    seed = 1984 if train else 2020
    np.random.seed(seed=seed)

    num_data, num_class, input_dim = 100, 3, 2
    data_size = num_class * num_data
    # data: (300, 2), label: (300,)
    x = np.zeros((data_size, input_dim), dtype=np.float32)
    t = np.zeros((data_size, num_class), dtype=np.int32)

    for cls in range(num_class):
        for i in range(num_data):
            rate = i / num_data
            radius = 1.0 * rate
            # theta: [class value] + [data value] + [turbulance]
            theta = (4.0 * cls)  + (4.0 * rate) + (0.2 * np.random.randn())
            # points are aligned with respect to classes
            ix = num_data * cls + i
            # (rsin(a), rcos(a))
            x[ix] = np.array([radius * np.sin(theta),
                              radius * np.cos(theta)]).flatten()
            # The answer is the class classification
            t[ix] = np.eye(num_class)[cls]

    # Shuffle
    if shuffle:
        indices = np.random.permutation(num_data * num_class)
        x = x[indices]
        t = t[indices]

    return x, t

def one_hot(t: int, options: int) -> np.ndarray:
    return np.eye(options)[t]
