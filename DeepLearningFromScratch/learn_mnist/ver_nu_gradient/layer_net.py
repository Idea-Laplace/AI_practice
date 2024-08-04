import os, sys
sys.path.append(os.pardir)
from dataclasses import dataclass, field
import numpy as np
import basic_functions as bf

@dataclass
class TwoLayerNet:
    input_size: int
    hidden_size: int
    output_size: int
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
    
    def predict(self, img_batch: np.ndarray) -> np.ndarray:
        w1, w2 = self.params['w1'], self.params['w2']
        b1, b2 = self.params['b1'], self.params['b2']

        intermediate = bf.relu(np.dot(img_batch, w1) + b1)
        raw_output = np.dot(intermediate, w2) + b2

        output = np.zeros_like(raw_output)
        for i in range(raw_output.shape[0]):
            output[i] = bf.softmax(raw_output[i])
        return output
    
    def loss(self, img_batch: np.ndarray, idx_batch: np.ndarray) -> float:
        output = self.predict(img_batch)
        return bf.cross_entropy_error(output, idx_batch)

    def accuracy(self, img_batch: np.ndarray, idx_batch: np.ndarray) -> float:
        output = self.predict(img_batch)
        prediction = output.argmax(axis=1)
        answer = idx_batch.argmax(axis=1)

        accuracy = np.sum(prediction == answer) / float(answer.shape[0])
        return accuracy
    
    def gradient(self, img_batch: np.ndarray, idx_batch: np.ndarray) -> dict:
        assert img_batch.shape[1] == self.params['w1'].shape[0],\
            'The size of the column of the input array should be accord with that of the row the layer array'

        loss_params = lambda param: self.loss(img_batch, idx_batch)
        grads = {}
        
        for key in self.params:
            grads[key] = bf.numerical_gradient(loss_params, self.params[key])

        return grads
    
    def learning(self, img_batch: np.ndarray, idx_batch: np.ndarray, lr=0.01):
        predict = self.predict(img_batch)
        loss = self.loss(img_batch, idx_batch)
        accuracy = self.accuracy(img_batch, idx_batch)

        print(f"Softmax: {predict.argmax(axis=1)}")
        print(f"Idx    : {idx_batch.argmax(axis=1)}")
        print(f"loss   : {loss: .5f}")
        print(f"accuracy: {100*accuracy: .2f}%")

        grads = self.gradient(img_batch, idx_batch)
        for key in self.params:
            self.params[key] -= lr * grads[key]

        
        
