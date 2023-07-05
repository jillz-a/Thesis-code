#Creates mock data files with knwon properties to test the behaviour of the models

import pandas as pd
import numpy as np


def simple_linear(num_sensors, output):
    """Generates test and train data sets where the output should be the sum of the inputs

    Args:
        num_sensors (int, optional): amount of inputs Defaults to 14.

    Returns:
        list, int : list of input values and corresponding output value
    """

    input = [np.random.uniform(0, output) for _ in range(num_sensors - 1)]
    input.append(output - sum(input))
    np.random.shuffle(input)

    
    
    return input, output


if __name__ == '__main__':
    n_train = 100
    n_test = 10
    num_sensors = 14

    for i in range(n_train):

        input, output = simple_linear(num_sensors, i)

        with open('verification_set/train/ver-train-{0:0=3d}.txt'.format(output), 'w') as f:
            f.write(" ".join(map(str, input)))

    for i in range(n_test):

        input, output = simple_linear(num_sensors, i)

        with open('verification_set/test/ver-test-{0:0=3d}.txt'.format(output), 'w') as f:
            f.write(" ".join(map(str, input)))

    
