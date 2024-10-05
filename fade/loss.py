import numpy as np

class Loss():
    def loss(self, y, y_pred):
        pass

    def df(self, y, y_pred):
        pass


class CrossEntropy(Loss):
    def loss(self, y, y_pred):
        nloss = -np.sum(y * np.log(y_pred + 1e-15), axis=1)
        return np.mean(nloss)
    
    def df(self, y, y_pred):
        return -y / (y_pred + 1e-15) 