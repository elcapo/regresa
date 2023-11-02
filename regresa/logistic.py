import numpy as np
import math, copy
from regresa.utils import numpyize_list

def sigmoid(z):
    """
    Compute the sigmoid of z. In other words, compute 1 / (1 + e**(-z)).

    Arguments:
        z (ndarray (m, )): one dimensional vector with the input values

    Returns:
        (ndarray (m, )): vector with the dimension of z and the result of the computation
    """
    z = numpyize_list(z)
    return 1 / (1 + np.exp(-z))

def predict(x, w, b):
    """
    Apply a given set of coefficients to the input to predict an output.

    Arguments:
        x (ndarray (m, n)): input values where the regression will be computed
        w (ndarray (n, )): weights for each of the x components
        b (scalar): additional weight for the regression

    Return:
        f_wb (ndarray (m, )): evaluation of the logistic regression for each value of x
    """
    x = numpyize_list(x)
    m = x.shape[0]
    f_wb = np.zeros(m)
    for i in range(m):
        z = (x[i] @ w) + b
        f_wb[i] = sigmoid(z)
    return f_wb

def loss(x, y, w, b, lambde=0):
    """
    Compute the loss of a set of examples.

    Arguments:
        x (ndarray (m, n)): input values where the regression will be computed
        y (ndarray (m, )): vector with boolean tags for each element in x
        w (ndarray (n, )): weights for each of the x components
        b (scalar): additional weight for the regression
        lambde (scalar): factor of regularization

    Returns:
        (ndarray (m, )): loss for each of the given examples
    """
    m = x.shape[0]
    n = w.shape[0]
    f_wb = predict(x, w, b)

    # standard loss
    l = np.zeros(m)
    for i in range(m):
        l[i] = (-y[i]) * np.log(f_wb[i]) - (1 - y[i]) * np.log(1 - f_wb[i])
    
    # regularization loss
    lr = (w @ w) / (2*m)
    return l + (lambde*lr)

def cost(x, y, w, b, lambde=0):
    """
    Compute the cost of a given set of components for w and b.

    Arguments:
        x (ndarray (m, n)): input values where the regression will be computed
        y (ndarray (m, )): vector with boolean tags for each element in x
        w (ndarray (n, )): weights for each of the x components
        b (scalar): additional weight for the regression
        lambde (scalar): factor of regularization

    Returns:
        (scalar): total cost for the given set of weights
    """
    m = x.shape[0]
    return sum(loss(x, y, w, b, lambde)) / m

def cost_gradient(x, y, w, b, lambde=0):
    """
    Compute the gradient of the cost of a given set of coefficients (w and b).

    Arguments:
        x (ndarray (m, n)): input values where the regression will be computed
        y (ndarray (m, )): vector with boolean tags for each element in x
        w (ndarray (n, )): weights for each of the x components
        b (scalar): additional weight for the regression
        lambde (scalar): factor of regularization

    Returns:
        (ndarray (n, )): gradient of the cost for the given set of weights w
        (scalar): gradient of the cost for the given weight b
    """
    m = x.shape[0]
    n = len(w)
    f_wb = predict(x, w, b)

    # standard cost
    dj_dw = np.zeros((n,))
    for i in range(n):
        dj_dw[i] = np.sum((f_wb - y) @ x[:,i]) / m
    dj_db = np.sum(f_wb - y) / m

    # regularized cost
    dj_dw += (lambde * w) / m
    return dj_dw, dj_db

def gradient_descent(x, y, w_in, b, alpha, iterations):
    """
    Compute a gradient descent.

    Arguments:
        x (ndarray (m, n)): input values where the regression will be computed
        y (ndarray (m, )): vector with boolean tags for each element in x
        w_in (ndarray (n, )): weights for each of the x components
        b (scalar): additional weight for the regression
        alpha (scalar): learning rate
        iterations (scalar): number of iterations to run

    Returns:
        (ndarray (n, )): weights for each feature after the iterations
        (scalar): additional scalar weight
    """
    w = copy.deepcopy(w_in)
    for i in range(iterations):
        dj_dw, dj_db = cost_gradient(x, y, w, b)
        w = w - (alpha * dj_dw)
        b = b - (alpha * dj_db)
    return w, b

x = [[x] for x in range(-10, 11)]
y = predict(x, [1], 0)
