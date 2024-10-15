import numpy as np

class Optimizer():
    def __init__(self, l_rate = 0.01):
        self.l_rate = l_rate

    def __call__(self, w, df):
        pass


class Default(Optimizer):
    def __init__(self, l_rate = 0.01):
        super().__init__(l_rate)

    def __call__(self, w, df):
        return w - self.l_rate * df
    

class Momentum(Optimizer):
    def __init__(self, l_rate = 0.01, b = 0.9):
        super().__init__(l_rate)

        self.b = b

        self.v = None

    def __call__(self, w, df):
        if self.v is None:
            self.v = np.zeros_like(w)
        self.v = self.l_rate * df - self.b * self.v 
        return w - self.v


class Adagrad(Optimizer):
    def __init__(self, l_rate = 0.01, e = 1e-9):
        super().__init__(l_rate)

        self.e = e

        self.g = None

    def __call__(self, w, df):
        if self.g is None:
            self.g = np.zeros_like(w)
        self.g += np.square(df)
        return w - self.l_rate * df / np.sqrt(self.g + self.e) 


class RMSProp(Optimizer):
    def __init__(self, l_rate = 0.01, b = 0.9, e = 1e-9):
        super().__init__(l_rate)

        self.b = b
        self.e = e

        self.g = None

    def __call__(self, w, df):
        if self.g is None:
            self.g = np.zeros_like(w)

        self.g = self.b * self.g + (1 - self.b) * np.square(df)
        return w - self.l_rate * df / np.sqrt(self.g + self.e) 


class Adam(Optimizer):
    def __init__(self, l_rate = 0.01, b1 = 0.9, b2 = 0.999, e = 1e-9):
        super().__init__(l_rate)

        self.b1 = b1
        self.b2 = b2
        self.e = e
        self.t = 0

        self.v = None
        self.g = None

    def __call__(self, w, df):
        self.t += 1

        if self.v is None:
            self.v = np.zeros_like(df)
            self.g = np.zeros_like(df)

        self.v = self.b1 * self.v + (1 - self.b1) * df
        self.g = self.b2 * self.g + (1 - self.b2) * np.square(df)

        v_corrected = self.v / (1 - self.b1**self.t)
        g_corrected = self.g / (1 - self.b2**self.t)

        return w - self.l_rate * v_corrected / (np.sqrt(g_corrected) + self.e) 
    

class AdamW(Optimizer):
    def __init__(self, l_rate = 0.01, b1 = 0.9, b2 = 0.999, e = 1e-9, wdecay = 1e-2):
        super().__init__(l_rate)

        self.b1 = b1
        self.b2 = b2
        self.e = e
        self.wdecay = wdecay
        self.t = 0
        self.v = None
        self.g = None

    def __call__(self, w, df):
        self.t += 1
        
        if self.v is None:
            self.v = np.zeros_like(df)
            self.g = np.zeros_like(df)

        self.v = self.b1 * self.v + (1 - self.b1) * df
        self.g = self.b2 * self.g + (1 - self.b2) * np.square(df)

        v_corrected = self.v / (1 - self.b1**self.t)
        g_corrected = self.g / (1 - self.b2**self.t)

        w -= self.l_rate * v_corrected / (np.sqrt(g_corrected) + self.e)

        return w - self.l_rate * self.wdecay * w
