import numpy as np
import math, copy
from regresa.utils import numpyize_list

def predict(X, w, b):
    """
    Apply a given set of coefficients to the input to predict an output.

    Arguments:
        X (ndarray (m, n)): input values where the regression will be computed
        w (ndarray (n, )): weights for each of the features
        b (scalar): biased weight for the regression

    Return:
        f_wb (ndarray (m, )): evaluation of the linear regression for each value of x
    """
    X = numpyize_list(X)
    m = X.shape[0]
    f_wb = np.zeros(m)
    for i in range(m):
        f_wb[i] = (X[i] @ w) + b
    return f_wb

def loss(X, y, w, b):
    """
    Compute the loss of a set of examples.

    Arguments:
        X (ndarray (m, n)): input values where the regression will be computed
        y (ndarray (m, )): vector with boolean tags for each example
        w (ndarray (n, )): weights for each of the features
        b (scalar): biased weight for the regression

    Returns:
        (ndarray (m, )): loss for each of the given examples
    """
    m = X.shape[0]
    n = w.shape[0]
    f_wb = predict(X, w, b)

    l = np.zeros(m)
    for i in range(m):
        l[i] = (f_wb[i] - y[i])**2    
    return l

def cost(X, y, w, b, lambde=0):
    """
    Compute the cost for a given set of examples.

    Arguments:
        X (ndarray (m, n)): input values where the regression will be computed
        y (ndarray (m, )): vector with boolean tags for each example
        w (ndarray (n, )): weights for each of the features
        b (scalar): biased weight for the regression
        lambde (scalar): factor of regularization

    Returns:
        (scalar): total cost for the given set of weights
    """
    m = X.shape[0]
    prediction_cost = sum(loss(X, y, w, b)) / (2*m)
    regularization = lambde * np.dot(w, w) / (2*m)
    return prediction_cost + regularization

def cost_gradient(X, y, w, b, lambde=0):
    """
    Compute the gradient of the cost for a given set of examples.

    Arguments:
        X (ndarray (m, n)): input values where the regression will be computed
        y (ndarray (m, )): vector with boolean tags for each example
        w (ndarray (n, )): weights for each of the features
        b (scalar): biased weight for the regression
        lambde (scalar): factor of regularization

    Returns:
        (ndarray (n, )): gradient of the cost for the given set of weights w
        (scalar): gradient of the cost for the given weight b
    """
    m = X.shape[0]
    n = len(w)
    f_wb = predict(X, w, b)

    # standard cost
    dj_dw = np.zeros((n,))
    for i in range(n):
        dj_dw[i] = np.sum((f_wb - y) @ X[:,i]) / m
    dj_db = np.sum(f_wb - y) / m

    # regularized cost
    dj_dw += (lambde * w) / m
    return dj_dw, dj_db

def gradient_descent(X, y, w_in, b, alpha, iterations):
    """
    Compute a gradient descent.

    Arguments:
        X (ndarray (m, n)): input values where the regression will be computed
        y (ndarray (m, )): vector with boolean tags for each example
        w (ndarray (n, )): weights for each of the features
        b (scalar): biased weight for the regression
        alpha (scalar): learning rate
        iterations (scalar): number of iterations to run

    Returns:
        (ndarray (n, )): weights for each feature after the iterations
        (scalar): additional scalar weight
    """
    w = copy.deepcopy(w_in)
    for i in range(iterations):
        dj_dw, dj_db = cost_gradient(X, y, w, b)
        w = w - (alpha * dj_dw)
        b = b - (alpha * dj_db)
    return w, b
