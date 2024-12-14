import numpy as np
from fade.layers import Layer

class ReLU(Layer):
    def __init__(self):
        super().__init__()

    def forward(self, x, training = False):
        self.out = np.maximum(0, x)
        return self.out
    
    def backward(self, df):
        return df * np.clip(self.out, 0, 1)


class Tanh(Layer):
    def __init__(self):
        super().__init__()

    def forward(self, x, training = False):
        self.out = (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))
        return self.out
    
    def backward(self, df):
        return df * (1 - np.square(self.out))
    

class Sigmoid(Layer):
    def __init__(self):
        super().__init__()

    def forward(self, x, training = False):
        self.out = 1 / (1 + np.exp(-x))
        return self.out
    
    def backward(self, df):
        return df * self.out * (1 - self.out)
    

class Softmax(Layer):
    def __init__(self):
        super().__init__()

    def forward(self, x, training = False):
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        self.out = exp_x / np.sum(exp_x, axis=-1, keepdims=True)
        return self.out

    def backward(self, df):
        gradients = np.zeros_like(df)
        
        for i, (out, dfi) in enumerate(zip(self.out, df)):
            jacobian = np.diag(out) - np.outer(out, out)
            gradients[i] = np.dot(jacobian, dfi)
        
        return gradients