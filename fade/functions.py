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

def one_hot_enc(y, n):
    ys = np.zeros((len(y), n))
    ys[np.arange(len(y)), y] = 1
    return ys

def split(X,Y, test_size=0.2):
    i = int(len(X) * test_size)
    return X[i :], X[: i], Y[i :], Y[: i]

def multiclass_accuracy(y, y_pred):
    n = len(y)
    return np.sum(np.argmax(y, axis=1) == np.argmax(y_pred, axis=1)) / n

def xavier_init(shape):
    if len(shape) == 2:
        d_in, d_out = shape
    elif len(shape) > 2:
        receptive_field_size = np.prod(shape[2:])
        d_in = shape[0] * receptive_field_size
        d_out = shape[1] * receptive_field_size
    
    std = np.sqrt(2.0 / (d_in + d_out))
    return np.random.normal(0, std, shape)


def he_init(shape):
    if len(shape) == 2:
        d_out = shape[1]
    elif len(shape) > 2:
        d_out = shape[1] * np.prod(shape[2:])
    std = np.sqrt(2.0 / d_out)
    return np.random.normal(0, std, size=shape)