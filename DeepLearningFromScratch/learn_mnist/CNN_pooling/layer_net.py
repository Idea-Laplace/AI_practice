import os, sys
sys.path.append(os.pardir)
from dataclasses import dataclass, field
from collections import OrderedDict
import numpy as np
from basic_layers import * 
from optimizers import *

@dataclass
class TwoLayerNet:
    input_size: int
    hidden_size: int
    output_size: int
    optimizer: Optimizer
    init_standard: float = 0.01

    params: dict = field(init=False)
    
    def __post_init__(self):
        self.params = {}
        self.params['w1'] = self.init_standard * \
            np.random.randn(self.input_size, self.hidden_size)
        self.params['b1'] = np.zeros(self.hidden_size)
        self.params['w2'] = self.init_standard * \
            np.random.randn(self.hidden_size, self.output_size)
        self.params['b2'] = np.zeros(self.output_size)
    
        self.layers = OrderedDict()
        self.layers['affine1'] = Affine(self.params['w1'], self.params['b1'])
        self.layers['relu'] = ReLu()
        self.layers['affine2'] = Affine(self.params['w2'], self.params['b2'])

        self.output_layer = SoftmaxLoss()


    def predict(self, img_batch: np.ndarray) -> np.ndarray:
        raw_output = img_batch
        for layer in self.layers.values():
            raw_output = layer.forward(raw_output)

        return raw_output
    
    def loss(self, img_batch: np.ndarray, idx_batch: np.ndarray) -> float:
        raw_output = self.predict(img_batch)
        return self.output_layer.forward(raw_output, idx_batch)

    def accuracy(self, img_batch: np.ndarray, idx_batch: np.ndarray) -> float:
        raw_output = self.predict(img_batch)
        prediction = raw_output.argmax(axis=1)
        answer = idx_batch.argmax(axis=1)

        accuracy = np.sum(prediction == answer) / float(answer.shape[0])
        return accuracy
    
    def gradient(self, img_batch: np.ndarray, idx_batch: np.ndarray) -> dict:
        assert img_batch.shape[1] == self.params['w1'].shape[0],\
            'The size of the column of the input array should be accord with that of the row the layer array'
        
        # Necessary instance variables for layer classes are specified by the loss function
        self.loss(img_batch, idx_batch)
        dout = self.output_layer.backward()

        reversed_layers = list(self.layers.values())
        reversed_layers.reverse()

        for layer in reversed_layers:
            dout = layer.backward(dout)

        grads = {}
        grads['w1'], grads['b1'] = self.layers['affine1'].dw, self.layers['affine1'].db
        grads['w2'], grads['b2'] = self.layers['affine2'].dw, self.layers['affine2'].db

        return grads

    
    def learning(self, img_batch: np.ndarray, idx_batch: np.ndarray, lr=0.001):
        predict = self.predict(img_batch)
        loss = self.loss(img_batch, idx_batch)
        accuracy = self.accuracy(img_batch, idx_batch)

        print(f"Predict : {predict.argmax(axis=1)}")
        print(f"Answer  : {idx_batch.argmax(axis=1)}")
        print(f"Loss    : {loss: .5f}")
        print(f"Accuracy: {100*accuracy: .2f}%")

        grads = self.gradient(img_batch, idx_batch)
        self.optimizer.update(self.params, grads)

        
        
