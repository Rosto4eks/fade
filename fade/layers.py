import numpy as np

class Layer():
    def __init__(self, optimizer = False):
        self.out = None
        self.optimizer = optimizer

    def forward(self, x):
        pass

    def backward(self, df):
        pass


class Linear(Layer):
    def __init__(self, n):
        super().__init__(True)
        self.n = n
        self.w = None
        self.b = None
        self.x = None

        self.woptimizer = None
        self.boptimizer = None

    def forward(self, x):
        if self.w is None:
            self.w = np.random.normal(0, np.sqrt(2 / (self.n + x.shape[1])), (self.n, x.shape[1]))
            self.b = np.zeros((1, self.n))
        self.x = x
        return np.dot(self.x, self.w.T) + self.b
    
    def backward(self, df):
        dout = np.dot(df, self.w)
        dw = np.dot(df.T, self.x)
        db = np.sum(df, axis=0)
        self.w -= self.woptimizer(dw)
        self.b -= self.boptimizer(db)
        return dout


class ReLU(Layer):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        self.out = np.maximum(0, x)
        return self.out
    
    def backward(self, df):
        return df * np.clip(self.out, 0, 1)
    

class Softmax(Layer):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        self.out = exp_x / np.sum(exp_x, axis=-1, keepdims=True)
        return self.out

    def backward(self, df):
        gradients = np.zeros_like(df)
        
        for i, (out, dfi) in enumerate(zip(self.out, df)):
            jacobian = np.diag(out) - np.outer(out, out)
            gradients[i] = np.dot(jacobian, dfi)
        
        return gradients
    
