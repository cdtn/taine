import numpy as np
from batchflow import plot



def plot_notifier_curve(ax, index, x, y, container, notifier, frequency, **kwargs):
    """ !!. """
    _ = index, x, container, notifier

    if len(y.shape) == 1 or y.shape[1] == 1:
        data = (np.arange(y.shape[0]) * frequency, y)
    else:
        data = [(np.arange(y.shape[0]) * frequency, y_values) for y_values in y.T]

    plot(data=data, mode='curve', axes=ax, **kwargs)
