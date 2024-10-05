import numpy as np 

def shuffle(X, Y):
    permutation = np.random.permutation(len(X))
    return X[permutation], Y[permutation]

def batches(X, Y, batch_size = 30):
    X, Y = shuffle(X, Y)
    x_len = len(X)
    count = int(np.ceil(x_len / batch_size))
    X_batches = [X[i * batch_size : (i+1) * batch_size] for i in range(count)]
    Y_batches = [Y[i * batch_size : (i+1) * batch_size] for i in range(count)]
    return zip(X_batches, Y_batches)

def one_hot_enc(y):
    ys = np.zeros((len(y), np.max(y) + 1))
    ys[np.arange(len(y)), y] = 1
    return ys

def split(X,Y, test_size=0.2):
    i = int(len(X) * test_size)
    return X[i :], X[: i], Y[i :], Y[: i]

def multiclass_accuracy(y, y_pred):
    n = len(y)
    return np.sum(np.argmax(y, axis=1) == np.argmax(y_pred, axis=1)) / n