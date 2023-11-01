from math import isclose
import numpy as np
from regresa.logistic import *

def test_sigmoid():
    assert sigmoid(0) == .5
    assert isclose(sigmoid(-10), 4.540e-05, rel_tol = 1e-4)
    assert isclose(sigmoid(10), 1, rel_tol = 1e-4)

def test_apply_regression():
    assert np.array_equal(
        apply_regression(np.array([[0]]), [0], 1),
        np.array([1 / (1 + np.exp(-1))])
    )

    assert np.array_equal(
        apply_regression(np.array([[1/2]]), [2], 0),
        np.array([1 / (1 + np.exp(-1))])
    )

    assert np.array_equal(
        apply_regression(np.array([[0], [0], [0]]), [0], 0),
        np.array([.5, .5, .5])
    )

def test_loss():
    x = np.array([[.5, 1.5], [1, 1], [1.5, .5], [3, .5], [2, 2], [1, 2.5]])
    y = np.array([0, 0, 0, 1, 1, 1])
    w = np.array([1,1])
    b = -3
    assert all(np.isclose(loss(x, y, w, b), [
        .31326169,
        .31326169,
        .31326169,
        .47407698,
        .31326169,
        .47407698
    ]))

def test_cost():
    x = np.array([[.5, 1.5], [1, 1], [1.5, .5], [3, .5], [2, 2], [1, 2.5]])
    y = np.array([0, 0, 0, 1, 1, 1])
    w = np.array([1, 1])
    b = -3
    assert isclose(cost(x, y, w, b), .3668667864055175)

def test_regularized_cost():
    np.random.seed(1)
    x = np.random.rand(5, 6)
    y = np.array([0, 1, 0, 1, 0])
    w = np.random.rand(x.shape[1]).reshape(-1,) - .5
    b = .5
    lambde = .7
    assert math.isclose(cost(x, y, w, b, lambde), .6850849138741673)

def test_cost_gradient():
    x = np.array([[.5, 1.5], [1, 1], [1.5, .5], [3, .5], [2, 2], [1, 2.5]])
    y = np.array([0, 0, 0, 1, 1, 1])
    w = np.array([2., 3.])
    b = 1.
    dj_dw, dj_db = cost_gradient(x, y, w, b)
    assert all(np.isclose(dj_dw, [
        .498333393278696,
        .49883942983996693
    ]))
    assert math.isclose(dj_db, .49861806546328574)

def test_regularized_cost_gradient():
    np.random.seed(1)
    x = np.random.rand(5, 3)
    y = np.array([0, 1, 0, 1, 0])
    w = np.random.rand(x.shape[1])
    b = .5
    lambde = .7
    dj_dw, dj_db = cost_gradient(x, y, w, b, lambde)
    assert all(np.isclose(dj_dw, [
        .17380012933994293,
        .32007507881566943,
        .10776313396851499,
    ]))
    assert math.isclose(dj_db, .341798994972791)

def test_gradient_descent():
    x = np.array([[.5, 1.5], [1, 1], [1.5, .5], [3, .5], [2, 2], [1, 2.5]])
    y = np.array([0, 0, 0, 1, 1, 1])
    w_in  = np.zeros_like(x[0])
    b_in = 0.
    alpha = .1
    iterations = 10000
    w, b = gradient_descent(x, y, w_in, b_in, alpha, iterations)
    assert all(np.isclose(w, [
        5.28123029,
        5.07815608,
    ]))
    assert math.isclose(b, -14.222409982019837)