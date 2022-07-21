import numpy as np


def variable_one_hot(shape):
    one_hot_weight_vector = np.zeros(shape)
    one_hot_weight_vector[..., 0] = 1
    return one_hot_weight_vector