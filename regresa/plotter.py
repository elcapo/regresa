import matplotlib.pyplot as plt
from matplotlib import colormaps
import numpy as np
from random import random
from regresa.utils import to_simple_list

def padding(x, percentage = 5):
    min_x, max_x = min(x), max(x)
    size = max_x - min_x
    size = size if size > 0 else 1
    padding_x = size * percentage / 100
    return min_x - padding_x, max_x + padding_x

def plot(x, y):
    x = to_simple_list(x)
    y = to_simple_list(y)
    figure, axe = plt.subplots()
    axe.plot(x, y, linewidth=2.0)
    axe.set(
        xlim=padding(x),
        ylim=padding(y),
    )
    plt.show()

def over_plot(x, multiple_y, labels = []):
    n = len(multiple_y)
    x = to_simple_list(x)

    figure, axe = plt.subplots()
    axe.set(
        xlim=padding(x),
        ylim=padding(np.array(multiple_y).reshape(-1, 1)),
    )

    colormap = colormaps['Blues'].resampled(n)

    for i in range(n):
        y = multiple_y[i]
        label = labels[i] if i < len(labels) else ''
        axe.plot(x, y, linewidth=2.0, color=colormap(i), label=label)
    
    plt.legend(loc="upper left")
    plt.show()
