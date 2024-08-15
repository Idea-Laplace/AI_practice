import os, sys
sys.path.append(os.pardir)
from collections import OrderedDict
import numpy as np
from basic_layers import * 
from convolution_and_pooling import *
from optimizers import *


class CNN:
    def __init__(self, batch_shape: tuple, optimizer: Optimizer=Adam(0.001), dropout=False):
        self.convolutions = OrderedDict()
        self.convolutions['normalizer0'] = Normalizer()
        self.convolutions['filter1'] = Convolution(batch_shape, (16, 1, 3, 3))
        self.convolutions['normalizer1'] = Normalizer()
        self.convolutions['relu1'] = ReLu()
        self.convolutions['filter2'] = Convolution((batch_shape[0], 16, 26, 26), (16, 16, 3, 3))
        self.convolutions['normalizer2'] = Normalizer()
        self.convolutions['relu2'] = ReLu()
        self.convolutions['pooling16'] = Pooling(2, 2)
        self.convolutions['filter3'] = Convolution((batch_shape[0], 16, 23, 23), (32, 16, 3, 3))
        self.convolutions['normalizer3'] = Normalizer()
        self.convolutions['relu3'] = ReLu()
        self.convolutions['filter4'] = Convolution((batch_shape[0], 32, 21, 21), (32, 32, 3, 3))
        self.convolutions['normalizer4'] = Normalizer()
        self.convolutions['relu4'] = ReLu()
        self.convolutions['pooling32'] = Pooling(2, 2)
        '''
        #self.convolutions['filter5'] = Convolution((batch_shape[0], 16, 18, 18), (64, 32, 3, 3))
        #self.convolutions['relu5'] = ReLu()
        #self.convolutions['filter6'] = Convolution((batch_shape[0], 64, 16, 16), (64, 64, 3, 3))
        #self.convolutions['relu6'] = ReLu()
        #self.convolutions['pooling64'] = Pooling(2, 2)
        '''
        

        self.inter = 18
        self.affines = OrderedDict()
        self.affines['affine1'] = Affine((self.inter * self.inter * 32, 100), 100)
        self.affines['normalizer1'] = Normalizer()
        self.affines['relu'] = ReLu()
        #self.affines['dropout1'] = DropOut(0.3)
        self.affines['affine2'] = Affine((100, 10), 10)
        self.affines['normalizer2'] = Normalizer()
        #self.affines['dropout2'] = DropOut(0.3)

        self.outlayer = SoftmaxLoss()

        self.optimizer = optimizer

        self.params = {}
        self.params['filter1_w'] = self.convolutions['filter1'].w
        self.params['filter1_b'] = self.convolutions['filter1'].b
        self.params['filter2_w'] = self.convolutions['filter2'].w
        self.params['filter2_b'] = self.convolutions['filter2'].b
        self.params['filter3_w'] = self.convolutions['filter3'].w
        self.params['filter3_b'] = self.convolutions['filter3'].b
        self.params['filter4_w'] = self.convolutions['filter4'].w
        self.params['filter4_b'] = self.convolutions['filter4'].b
        '''
        self.params['filter5_w'] = self.convolutions['filter5'].w
        self.params['filter5_b'] = self.convolutions['filter5'].b
        self.params['filter6_w'] = self.convolutions['filter6'].w
        self.params['filter6_b'] = self.convolutions['filter6'].b
        '''
        self.params['affine1_w'] = self.affines['affine1'].w
        self.params['affine1_b'] = self.affines['affine1'].b
        self.params['affine2_w'] = self.affines['affine2'].w
        self.params['affine2_b'] = self.affines['affine2'].b

    
    def predict(self, x: np.ndarray) -> np.ndarray:
        assert x.ndim == 4, 'This version should accept 4D arry, (num_img, channel, row, column)'

        output = x
        for conv_layer in self.convolutions.values():
            output = conv_layer.forward(output)
        
        output = output.reshape(output.shape[0], -1)

        for affine_layer in self.affines.values():
            output = affine_layer.forward(output)

        return output
    
    def loss(self, x: np.ndarray, t: np.ndarray) -> float:
        output = self.predict(x)
        loss_value = self.outlayer.forward(output, t)
        return loss_value
    
    def accuracy(self, img_batch: np.ndarray, idx_batch: np.ndarray) -> float:
        raw_output = self.predict(img_batch)
        prediction = raw_output.argmax(axis=1)
        answer = idx_batch.argmax(axis=1)

        accuracy = np.sum(prediction == answer) / float(answer.shape[0])
        return accuracy
    
    def gradient(self, img_batch: np.ndarray, idx_batch: np.ndarray) -> dict:
        # Necessary instance variables for layer classes are specified by the loss function
        self.loss(img_batch, idx_batch)
        dout = self.outlayer.backward()

        reversed_affines = list(self.affines.values())
        reversed_affines.reverse()
        reversed_convolutions = list(self.convolutions.values())
        reversed_convolutions.reverse()

        for layer in reversed_affines:
            dout = layer.backward(dout)
        
        dout = dout.reshape(dout.shape[0], -1, self.inter, self.inter)
        for layer in reversed_convolutions:
            dout = layer.backward(dout)

        grads = {}
        for key in self.affines:
            if key.startswith('affine'):
                grads[key + '_w'] = self.affines[key].dw 
                grads[key + '_b'] = self.affines[key].db

        for key in self.convolutions:
            if key.startswith('filter'):
                grads[key + '_w'] = self.convolutions[key].dw 
                grads[key + '_b'] = self.convolutions[key].db

        return grads

    
    def learning(self, img_batch: np.ndarray, idx_batch: np.ndarray):
        predict = self.predict(img_batch)
        loss = self.loss(img_batch, idx_batch)
        accuracy = self.accuracy(img_batch, idx_batch)

        print(f"Predict : {predict.argmax(axis=1)}")
        print(f"Answer  : {idx_batch.argmax(axis=1)}")
        print(f"Loss    : {loss: .5f}")
        print(f"Accuracy: {100*accuracy: .2f}%")

        grads = self.gradient(img_batch, idx_batch)

        self.optimizer.update(self.params, grads)

        '''
        for key in grads:
            if key.startswith('affine'):
                self.affines[key].w -= lr * grads[key][0]
                self.affines[key].b -= lr * grads[key][1]

            if key.startswith('filter'):
                self.convolutions[key].w -= lr * grads[key][0]
                self.convolutions[key].b -= lr * grads[key][1]
        '''
