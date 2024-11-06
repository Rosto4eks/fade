import numpy as np
from fade.functions import *

class Layer():
    def __init__(self, optimizer = False):
        self.out = None
        self.optimizer = optimizer

    def forward(self, x, training = False):
        pass

    def backward(self, df):
        pass


class Linear(Layer):
    def __init__(self, n, init_fn = he_init):
        super().__init__(True)
        self.n = n
        self.w = None
        self.b = None
        self.x = None
        self.init_fn = init_fn

        self.woptimizer = None
        self.boptimizer = None

    def forward(self, x, training = False):
        if self.w is None:
            self.w = self.init_fn((self.n, x.shape[1]))
            self.b = np.zeros((1, self.n))
        self.x = x
        return np.dot(self.x, self.w.T) + self.b
    
    def backward(self, df):
        dout = np.dot(df, self.w)
        dw = np.dot(df.T, self.x)
        db = np.sum(df, axis=0)
        self.w = self.woptimizer(self.w, dw)
        self.b = self.boptimizer(self.b, db)
        return dout


class ReLU(Layer):
    def __init__(self):
        super().__init__()

    def forward(self, x, training = False):
        self.out = np.maximum(0, x)
        return self.out
    
    def backward(self, df):
        return df * np.clip(self.out, 0, 1)

class tanh(Layer):
    def __init__(self):
        super().__init__()

    def forward(self, x, training = False):
        self.out = (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))
        return self.out
    
    def backward(self, df):
        return df * (1 - np.square(self.out))
    
class sigmoid(Layer):
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
    



class Conv(Layer):
    def __init__(self, kernel_count, kernel_size = 3, stride = 1, init_fn = he_init):
        super().__init__(True)
        self.kernel_count = kernel_count
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = kernel_size // 2
        self.init_fn = init_fn

        self.kernel = None
        self.bias = None
        self.woptimizer = None
        self.boptimizer = None

    def forward(self, x, training = False):
        while np.ndim(x) < 4:
            x = x[np.newaxis, :]

        self.n = x.shape[0]
        self.layers = x.shape[1]
        self.height = x.shape[2]
        self.width = x.shape[3]

        if self.kernel is None:
            self.kernel = self.init_fn((self.kernel_count, self.layers, self.kernel_size, self.kernel_size))
            self.bias = np.zeros((1, self.kernel_count, 1, 1))

        x = np.pad(x, ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)), mode='constant', constant_values=0)

        self.x = x

        h_new = (self.height - self.kernel_size + 2 * self.padding) // self.stride + 1
        w_new = (self.width - self.kernel_size + 2 * self.padding) // self.stride + 1
        out = np.zeros((self.n, self.kernel_count, h_new, w_new))

        for i in range(0, self.height, self.stride):
            for j in range(0, self.width, self.stride):
                einsum = np.einsum("nlhw,clhw->ncwh", x[:, :, i:i+self.kernel_size, j:j+self.kernel_size], self.kernel)
                out[:, :, i // self.stride, j // self.stride] = np.sum(einsum, axis=(2,3))

        out += self.bias

        return out

    def backward(self, df):
        dw = np.zeros(shape=(self.kernel_count, self.layers, self.kernel_size, self.kernel_size))
        dx = np.zeros(shape=(self.n, self.layers, self.height + 2 * self.padding, self.width + 2 * self.padding))
        db = np.sum(df, axis=(0, 2, 3)).reshape(1, self.kernel_count, 1, 1)

        for i in range(0, df.shape[2] * self.stride, self.stride):
            for j in range(0, df.shape[3] * self.stride, self.stride):
                x_slice = self.x[:, :, i:i+self.kernel_size, j:j+self.kernel_size]
                df_slice = df[:, :, i//self.stride, j//self.stride]

                # Compute dw
                dw += np.einsum("nlhw,nc->clhw", x_slice, df_slice)

                # Compute dx
                dx[:, :, i:i+self.kernel_size, j:j+self.kernel_size] += np.einsum("nc,clhw->nlhw", df_slice, self.kernel)

        self.kernel = self.woptimizer(self.kernel, dw)
        self.bias = self.boptimizer(self.bias, db)

        return dx[:, :, self.padding:self.height + self.padding, self.padding:self.width + self.padding]
            
                
class MaxPooing(Layer):
    def __init__(self, kernel_size = 2, stride = 2, padding = 0):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding


    def forward(self, x, training = False):
        while np.ndim(x) < 4:
            x = x[np.newaxis, :]

        self.x = x

        self.n = x.shape[0]
        self.layers = x.shape[1]
        self.height = x.shape[2]
        self.width = x.shape[3]

        new_height = (self.height + 2 * self.padding) // self.stride
        new_width = (self.width + 2 * self.padding) // self.stride

        out = np.zeros(shape=(self.n, self.layers, new_height, new_width))
        
        for i in range(0, self.height, self.stride):
            for j in range(0, self.width, self.stride):
                window = x[:, :, i:i+self.kernel_size, j:j+self.kernel_size]
                out[:, :, i // self.stride, j // self.stride] = np.max(window, axis=(2, 3))
                
        return out

    def backward(self, df):
        out = np.zeros(shape=(self.n, self.layers, self.height, self.width))
        
        for i in range(0, self.height, self.stride):
            for j in range(0, self.width, self.stride):
                window = self.x[:, :, i:i+self.kernel_size, j:j+self.kernel_size]
                mask = window == np.max(window, axis=(2, 3), keepdims=True)
                dw = window * mask * df[:, :, i // self.stride:i // self.stride+1, j // self.stride:j // self.stride+1]
                out[:, :, i:i+self.kernel_size, j:j+self.kernel_size] += dw

        return out

class Flatten(Layer):
    def __init__(self):
        super().__init__()
        self.shape = None
        pass

    def forward(self, x, training = False):
        self.shape = x.shape
        return np.reshape(x, (x.shape[0], -1))
    
    def backward(self, df):
        return np.reshape(df, self.shape)
    

class Dropout(Layer):
    def __init__(self, prob = 0.25):
        super().__init__()
        self.prob = prob
        self.mask = None

    def forward(self, x, training = False):
        if training:
            self.mask = np.random.binomial(1, 1 - self.prob, x.shape[1:]) / (1 - self.prob)
            return x * self.mask 
        else:
            return x

    def backward(self, df):
        return df * self.mask
    

class BatchNorm(Layer):
    def __init__(self):
        super().__init__(True)
        self.e = 1e-12
        self.w = None
        self.b = None

        self.woptimizer = None
        self.boptimizer = None

        self.mean = None
        self.std = None
        self.run_mean = None
        self.run_std = None
        self.run_coef = 0.9

        self.x_norm = None

    def forward(self, x, training = False):
        batch_size = x.shape[0]
        self.x = x
        if self.w is None:
            self.w = np.ones(shape=x.shape[1:])
            self.b = np.zeros(shape=x.shape[1:])
            self.run_mean = np.zeros(shape=x.shape[1:])
            self.run_std = np.zeros(shape=x.shape[1:])
        if training:
            self.mean = np.mean(x, axis=0)
            self.std = np.std(x, axis=0)

            self.run_mean = self.run_mean + (1 - self.run_coef) * self.mean
            self.run_std = self.run_std + (1 - self.run_coef) * self.std

            x = (x - self.mean) / (self.std + self.e)
            self.x_norm = x
        else:
            x = (x - self.run_mean) / (self.run_std + self.e)

        return self.w * x + self.b

    def backward(self, df):
        dw = np.sum(df * self.x_norm, axis=0)
        db = np.sum(df, axis=0)
        self.w = self.woptimizer(self.w, dw)
        self.b = self.boptimizer(self.b, db)

        dnorm = df * self.w
        N = self.x_norm.shape[0]
        
        dvar = np.sum(dnorm * (self.x - self.mean) * -0.5 * (self.std + self.e)**(-3), axis=0)
        dmean = np.sum(dnorm * -1 / np.sqrt(self.std + self.e), axis=0) + dvar * np.mean(-2 * (self.x - self.mean), axis=0)
        dx = (dnorm / np.sqrt(self.std + self.e)) + (dvar * 2 * (self.x - self.mean) / N) + (dmean / N)
        return dx

        