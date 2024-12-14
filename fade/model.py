from fade.layers import *
from fade.loss import *
from fade.optimizers import *
from fade.functions import *
import copy

class Network():
    def __init__(self, loss = "cross_entropy", optimizer = "adam"):
        self.layers: list[Layer] = []

        if loss == "cross_entropy":
            self.loss = CrossEntropy()
        else:
            raise Exception("invalid loss")
        

        if optimizer == "adam":
            self.optimizer = Adam()
        elif optimizer == "adamw":
            self.optimizer = AdamW()
        elif optimizer == "momentum":
            self.optimizer = Momentum()
        elif optimizer == "adagrad":
            self.optimizer = Adagrad()
        elif optimizer == "rmsprop":
            self.optimizer = RMSProp()
        elif optimizer == "default":
            self.optimizer = Default()
        else:
            raise Exception("invalid optimizer")
        
        self.out = None

    def add(self, layer: Layer):
        for i in range(layer.n_optimizers):
            layer.optimizers.append(copy.copy(self.optimizer)) 
        self.layers.append(layer)

    def forward(self, x, training = False):
        if np.ndim(x) == 1:
            x = x[np.newaxis, :]
        for layer in self.layers:
            x = layer.forward(x, training)
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
                self.out = self.forward(x, True)
                self.backprop(y)
                print(f"epoch {e + 1}/{epoch} batch: {j + 1}/{len(X) // batch_size} | loss: {self.loss.loss(y, self.out):.7f}", end='\r')

