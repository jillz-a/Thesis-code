#Creates mock data files with knwon properties to test the behaviour of the models

import pandas as pd
import numpy as np



def simple_linear(window, sensors, output):
    """Generates mock input data where all columns (sensors) are linear functions that all add up to a desired output.

    Args:
        window (int): sequence length of the input (amount of rows)
        sensors (int): feature size of the input (amount of columns)
        output (int): desired output that the input must add up to

    Returns:
        array, int: input array of size [window, sensors] of floats, the desired output
    """
    # Generate random coefficients for the linear functions
    coefficients = np.random.rand(sensors, window)
    
    # Normalize the coefficients to ensure the total sum is equal to the output
    coefficients /= np.sum(coefficients, axis=1, keepdims=True)
    
    # Generate the linear array
    input = np.zeros((window, sensors))
    for i in range(sensors):
        input[:, i] = np.arange(window) * coefficients[i]
    
    # Scale the linear array to match the desired output
    scaling_factor = output / np.sum(input)
    input *= scaling_factor

    
    return input, output


if __name__ == '__main__':
    n_train = 1000
    n_test = 10
    num_sensors = 14
    window = 30

    for i in range(1,n_train):

        input, output = simple_linear(window=window, sensors=num_sensors, output=i)
        
        np.savetxt('verification_set/train/ver-train-{0:0=3d}.txt'.format(output), input, fmt = '%.10f')
      

    for i in range(1,n_test):

        input, output = simple_linear(window=window, sensors=num_sensors, output=10*i)

        np.savetxt('verification_set/test/ver-test-{0:0=3d}.txt'.format(output), input, fmt = '%.10f')

       

    
