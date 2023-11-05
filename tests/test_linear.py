from math import isclose
import numpy as np
from regresa.linear import *

def test_predict():
    np.random.seed(1)
    X = np.random.rand(5, 6)
    y = np.array([0, 1, 0, 1, 0])
    w = np.random.rand(X.shape[1]).reshape(-1,)-.5
    b = .5
    lambde = .7
    assert all(np.isclose(predict(X, w, b), [
        .29687707,
        .55149312,
        .36038278,
        .7591659,
        -.011763,
    ]))

def test_loss():
    np.random.seed(1)
    X = np.random.rand(5, 6)
    y = np.array([0, 1, 0, 1, 0])
    w = np.random.rand(X.shape[1]).reshape(-1,)-.5
    b = .5
    lambde = .7
    assert all(np.isclose(loss(X, y, w, b), [
        .08813599335588754,
        .2011584220816214,
        .1298757447742463,
        .058001065685670254,
        .0001383681701169254,
    ]))

def test_regularized_cost():
    np.random.seed(1)
    X = np.random.rand(5, 6)
    y = np.array([0, 1, 0, 1, 0])
    w = np.random.rand(X.shape[1]).reshape(-1,)-.5
    b = .5
    lambde = .7
    assert isclose(cost(X, y, w, b, lambde), .07917239320214275)

def test_regularized_cost_gradient():
    np.random.seed(1)
    X = np.random.rand(5, 3)
    y = np.array([0, 1, 0, 1, 0])
    w = np.random.rand(X.shape[1])
    b = .5
    lambde = .7
    dj_dw, dj_db = cost_gradient(X, y, w, b, lambde)
    assert all(np.isclose(dj_dw, [
        .29653214748822276,
        .4911679625918033,
        .21645877535865857
    ]))
    assert math.isclose(dj_db, .6648774569425726)

def test_gradient_descent():
    X = np.array([[1.], [2.]])
    y = np.array([300., 500.])
    w_init = np.array([0])
    b_init = 0
    iterations = 10000
    alpha = .01
    w, b = gradient_descent(X, y, w_init, b_init, alpha, iterations)
    assert math.isclose(w[0], 199.99285075)
    assert math.isclose(b, 100.011567727362)
