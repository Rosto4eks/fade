from fade.layers import *
from fade.loss import *
from fade.optimizers import *
from fade.functions import *
import copy

class Network():
    def __init__(self, loss: Loss = CrossEntropy(), optimizer: Optimizer = Adam()):
        self.layers: list[Layer] = []
        self.loss = loss
        self.optimizer = optimizer
        self.out = None

    def add(self, layer: Layer):
        if layer.optimizer:
            layer.woptimizer = copy.copy(self.optimizer)
            layer.boptimizer = copy.copy(self.optimizer) 
        self.layers.append(layer)

    def forward(self, x):
        if np.ndim(x) == 1:
            x = x[np.newaxis, :]
        for layer in self.layers:
            x = layer.forward(x)
        return x
    
    def backprop(self, y):
        if np.ndim(y) == 1:
            y = y[np.newaxis, :]
        df = self.loss.df(y, self.out)
        for layer in reversed(self.layers):
            df = layer.backward(df)

    def train(self, X, Y, epoch=100, batch_size=30):
        for e in range(0, epoch):
            bcs = batches(X, Y, batch_size)
            for j, batch in enumerate(bcs):
                x, y = batch
                self.out = self.forward(x)
                self.backprop(y)
                print(f"epoch {e + 1}/{epoch} batch: {j + 1}/{len(X) // batch_size} | loss: {self.loss.loss(y, self.out):.7f}", end='\r')

