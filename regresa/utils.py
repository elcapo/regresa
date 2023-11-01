import numpy as np

def numpyize_list(l):
    if type(l) == list:
        return np.array(l)
    else:
        return l

def to_simple_list(x):
    x = numpyize_list(x)
    if len(x.shape) == 1:
        return x
    if len(x.shape) == 2 and x.shape[1] != 1:
        raise Exception('Only one dimensional matrices can be converted to lists')
    return x[:,0]