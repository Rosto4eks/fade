import numpy as np

class Optimizer():
    def __init__(self, l_rate = 0.01):
        self.l_rate = l_rate

    def __call__(self, df):
        pass


class Default(Optimizer):
    def __init__(self, l_rate = 0.01):
        super().__init__(l_rate)

    def __call__(self, df):
        return self.l_rate * df
    

class Momentum(Optimizer):
    def __init__(self, l_rate = 0.01, b = 0.9):
        super().__init__(l_rate)

        self.b = b

        self.v = 0

    def __call__(self, df):
        self.v = self.l_rate * df - self.b * self.v 
        return self.v


class Adagrad(Optimizer):
    def __init__(self, l_rate = 0.01, e = 1e-9):
        super().__init__(l_rate)

        self.e = e

        self.g = 0

    def __call__(self, df):
        self.g += np.square(df)
        return self.l_rate * df / np.sqrt(self.g + self.e) 


class RMSProp(Optimizer):
    def __init__(self, l_rate = 0.01, b = 0.9, e = 1e-9):
        super().__init__(l_rate)

        self.b = b
        self.e = e

        self.g = 0

    def __call__(self, df):
        self.g = self.b * self.g + (1 - self.b) * np.square(df)
        return self.l_rate * df / np.sqrt(self.g + self.e) 


class Adam(Optimizer):
    def __init__(self, l_rate = 0.01, b1 = 0.9, b2 = 0.999, e = 1e-9):
        super().__init__(l_rate)

        self.b1 = b1
        self.b2 = b2
        self.e = e

        self.v = 0
        self.g = 0

    def __call__(self, df):
        self.v = self.b1 * self.v + (1 - self.b1) * df
        self.g = self.b2 * self.g + (1 - self.b2) * np.square(df)
        return self.l_rate * self.v / np.sqrt(self.g + self.e) 
