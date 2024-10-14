from fade.layers import *
from fade.loss import *
from fade.optimizers import *
from fade.functions import *


class Network():
    def __init__(self, loss: Loss = CrossEntropy, optimizer: Optimizer = Adam):
        self.layers: list[Layer] = []
        self.loss = loss()
        self.optimizer = optimizer
        self.out = None

    def add(self, layer: Layer):
        if layer.optimizer:
            layer.woptimizer = self.optimizer()
            layer.boptimizer = self.optimizer() 
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
        x_train, x_val, y_train, y_val = split(X, Y)
        for e in range(1, epoch + 1):
            bcs = batches(x_train, y_train, batch_size)
            j = 0
            for x, y in bcs:
                self.out = self.forward(x)
                self.backprop(y)
                print(f"batch: {j} / {len(x_train) / batch_size} | loss: {self.loss.loss(y, self.out):.7f}", end='\r')
                j += 1
            print("")
            y_pred = self.forward(x_val)
            print(f"loss: {self.loss.loss(y_val, y_pred):.7f}  epoch {e}/{epoch}")
            print("")

