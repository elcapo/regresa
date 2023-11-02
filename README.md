# Regresa

**Regresa** is a Python package where I implemented my own versions of the logistic regression algorithms presented by [Andrew Ng](https://www.andrewng.org) in his [Supervised Machine Learning: Regression and Classification](https://www.coursera.org/learn/machine-learning?specialization=machine-learning-introduction) course.

My motivations were:

1. to have reusable implementation of the algorithms for learning purposes,
2. to write the algorithms avoiding nested loops one single function for readability,
3. to add tests to the implementations so I could play refactoring them.

## Installation

**Regresa** is written with [Poetry](https://python-poetry.org). The following instructions should be sufficient for you to start using it.

```bash
git clone https://github.com/elcapo/regresa.git
cd regresa
poetry install
```

> Note that you'll need to install **git**, **python** and **poetry** to get this working.

## Usage

Once installed, use **Poetry**'s shell to interact with the package.

```bash
poetry shell
```

### Logistic

The **logistic** module offers functions to compute a binary classification given a set of examples with one or more features:

- [sigmoid](#sigmoid): compute the sigmoid of z; in other words, compute `1 / (1 + e**(-z))`
- [predict](#predict): apply a given set of coefficients to the input to predict an output
- [loss](#loss): compute the loss of a set of examples
- [cost](#cost): compute the cost of a given set of components for w and b
- [cost_gradient](#cost-gradient): compute the gradient of the cost of a given set of coefficients
- [gradient_descent](#gradient-descent): compute a gradient descent

These functions can be imported one by one:

```python
from regresa.logistic import sigmoid

# ... and then use them directly by their names
sigmoid(.5)
```

... or all at once:

```python
from regresa import logistic

# ... and then use them directly by their names prefixed with logistic
logistic.sigmoid(.5)
```

#### Sigmoid

```python
from regresa.logistic import sigmoid

help(sigmoid)
```

```
sigmoid(z)
    Compute the sigmoid of z. In other words, compute 1 / (1 + e**(-z)).
    
    Arguments:
        z (ndarray (m, )): one dimensional vector with the input values
    
    Returns:
        (ndarray (m, )): vector with the dimension of z and the result of the computation
```

This function accepts scalars as input. If a scalar is given, a scalar is also returned.

```python
sigmoid(0) # 0.5
sigmoid(9**9) # 1.0
```

The function also accepts lists of numbers and Numpy arrays as input. In those cases, a Numpy array with the same dimension of the input is returned.

```python
sigmoid([0, 9**9]) # array([0.5, 1. ])
```

In combination with the `plot` method from the `plotter` module, you can easily have a glimpse on how the function looks like.

```python
from regresa.logistic import sigmoid
from regresa.plotter import plot

x = [x for x in range(-10, 10 + 1)]
y = sigmoid(x)

plot(x, y)
```

![Sigmoid function plotted from x = -10 to x = +10](assets/sigmoid_plot.png)

#### Predict

```python
from regresa.logistic import predict

help(predict)
```

```
predict(x, w, b)
    Apply a given set of coefficients to the input to predict an output.
    
    Arguments:
        x (ndarray (m, n)): input values where the regression will be computed
        w (ndarray (n, )): weights for each of the x components
        b (scalar): additional weight for the regression
    
    Return:
        f_wb (ndarray (m, )): evaluation of the logistic regression for each value of x
```

In combination with the `plot` method from the `plotter` module, you check how a logistic regression graph changes with different weights.

```python
from regresa import plotter, logistic

x = [[x/10] for x in range(-100, 110, 1)]
multiple_y = [logistic.predict(x, [d/10], 0) for d in range(0, 12, 2)]
labels = ['w = {}'.format(d/10) for d in range(0, 12, 2)]

plotter.over_plot(x, multiple_y, legends)
```

![Logistic regression for weights increasing in factors of 0.2](assets/predict_plot.png)


#### Loss

```python
from regresa.logistic import loss

help(loss)
```

```
loss(x, y, w, b, lambde=0)
    Compute the loss of a set of examples.
    
    Arguments:
        x (ndarray (m, n)): input values where the regression will be computed
        y (ndarray (m, )): vector with boolean tags for each element in x
        w (ndarray (n, )): weights for each of the x components
        b (scalar): additional weight for the regression
        lambde (scalar): factor of regularization
    
    Returns:
        (ndarray (m, )): loss for each of the given examples
```

The loss function is a convenience that helps us debugging the cost function, as its the part of the cost that corresponds to each example. It's given by:

```math
j_i = -y_i log(f_{\vec{w},i}(\vec{x_i})) - (1 - y_i) log(1 - f_{\vec{w},i}(\vec{x_i})) - \frac{\lambda}{2m} \vec{w} \cdot \vec{w}
```

#### Cost

```python
from regresa.logistic import cost

help(cost)
```

```
cost(x, y, w, b, lambde=0)
    Compute the cost of a given set of components for w and b.
    
    Arguments:
        x (ndarray (m, n)): input values where the regression will be computed
        y (ndarray (m, )): vector with boolean tags for each element in x
        w (ndarray (n, )): weights for each of the x components
        b (scalar): additional weight for the regression
        lambde (scalar): factor of regularization
    
    Returns:
        (scalar): total cost for the given set of weights
```

Thanks to the convenience loss function described above, the cost expression can be simplified as follows.

```math
J(\vec{w}, b) = \frac{1}{m} \sum_{i=1}^{m} j_i
```

#### Cost Gradient

```python
from regresa.logistic import cost_gradient

help(cost_gradient)
```

```
cost_gradient(x, y, w, b, lambde=0)
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
```

The cost gradient is given by the partial derivates of the cost described above with respect the coefficients $w_j$ and $b$.

```math
\frac{\partial J(\vec{w}, b)}{\partial w_j} = \frac{1}{m} \sum_{i=1}^{m} [f_{\vec{w},b}(\vec{x}_i) - y_i] x_j + \frac{\lambda}{m} w_j
```

```math
\frac{\partial J(\vec{w}, b)}{\partial b} = \frac{1}{m} \sum_{i=1}^{m} [f_{\vec{w},b}(\vec{x}_i) - y_i]
```

#### Gradient Descent

```python
from regresa.logistic import gradient_descent

help(gradient_descent)
```

```
gradient_descent(x, y, w_in, b, alpha, iterations)
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
```

The gradient descent iteratively updates the coefficients `w` and `b` according to these expressions:

```math
w_j^i = w_j^{i-1} - \alpha \frac{\partial J(\vec{w}, b)}{\partial w_j}
```

```math
b^i = b^{i-1} - \alpha \frac{\partial J(\vec{w}, b)}{\partial b}
```

> Note that the superscript in $w_j^i$ does not represent a power. Instead, it express that this is the value of $w_j$ that corresponds with the iteration $i$.

## Tests

To run the tests, use **PyTest** from your shell.

```bash
pytest -v
```

![Example of result of running the test suite](assets/screenshot_of_the_first_testuite.png)