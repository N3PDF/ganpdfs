# Some useful Functions

import numpy as np
from math import gcd


def factorize_number(number):
    """factorize_number using Pollard's rho
    algorithm.

    Parameters
    ----------
    number :
        number
    """

    factors = []

    def get_factor(number):
        x = 2
        factor = 1
        x_fixed = 2
        cycle_size = 2

        while factor == 1:
            for count in range(cycle_size):
                if factor > 1:
                    break
                x = (x * x + 1) % number
                factor = gcd(x - x_fixed, number)
            cycle_size *= 2
            x_fixed = x
        return factor

    while number > 1:
        new_number = get_factor(number)
        factors.append(new_number)
        number //= new_number

    return factors


def construct_cnn(number, nb_layer):
    """construct_cnn.

    Parameters
    ----------
    factorized :
        factorized
    nb_layer :
        nb_layer
    """

    factorized = factorize_number(number)
    size = len(factorized)
    cnn_dim = sorted(factorized)
    if size < nb_layer:
        for _ in range(1, nb_layer - size + 1):
            cnn_dim.append(1)
    elif size > nb_layer:
        cnn_dim = cnn_dim[:nb_layer - 1]
        new_elem = np.prod(np.array(cnn_dim))
        cnn_dim.append(new_elem)
    else:
        pass
    return cnn_dim
