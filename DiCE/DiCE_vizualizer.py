#%% import dependencies
import sys
import os

# Get the absolute path of the project directory
project_path = os.path.dirname(os.path.abspath(os.path.join((__file__), os.pardir)))
# Add the project directory to sys.path if it's not already present
if project_path not in sys.path:
    sys.path.append(project_path)

from variables import *
import glob
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from trend_classifier import Segmenter

#definitions
def plot_segments(data, n):
    """Plots trend segments over original data

    Args:
        segments (Segment): Segment object containing information for linear trends
        n (int): Number of samples in a window
    """
    trend = Segmenter(list(np.arange(len(data))), data.to_list(), n=n)
    trend.calculate_segments()
    df = trend.segments.to_dataframe()
    for i in range(len(df)):
        start = df['start'][i]
        stop = df['stop'][i]
        slope = df['slope'][i]
        offset = df['offset'][i]

        x = np.arange(start=start, stop=stop)
        y = x*slope + offset

        ax.plot(x,y)

#%% import files
TRAINDATASET = f'data/{DATASET}/min-max/train'
TESTDATASET = f'data/{DATASET}/min-max/test'

#counterfactual input samples
result_path = os.path.join(project_path, 'DiCE/results', DATASET)
samples = glob.glob(os.path.join(result_path, '*.csv'))  # Get a list of all file paths in the folder
samples.sort()

#Original inputs
file_paths = glob.glob(os.path.join(project_path, TESTDATASET, '*.txt'))  # Get a list of all file paths in the folder
file_paths.sort() 

#%% Plot counterfacutal dataframe
for i, sample in enumerate(samples[0:1]):

    #Counterfactuals
    cf_total = pd.read_csv(sample)
    cf_RUL = cf_total['RUL']
    cf_total = cf_total.drop('RUL', axis=1)


    fig, axes = plt.subplots(nrows=2, ncols=7, sharex=True,
                                            figsize=(25, 8))

    sensor = 0
    m = [2,3,4,7,8,9,11,12,13,14,15,17,20,21] #useful sensors
    #go over every sensor
    for ax in axes.ravel():

        for i in range(len(cf_total)):
            cf_df = cf_total.iloc[[i]]
            cf_df = cf_df.values.reshape(30,14)
            cf_df = pd.DataFrame(cf_df)

            counter = cf_df[sensor]
            ax.plot(range(len(counter)), counter, label=f'CF {i + 1}: RUL = {cf_RUL.iloc[i]}', linestyle='--')
            
            plot_segments(counter, n=29)

        #Original inputs
        df_orig = pd.read_csv(file_paths[i], sep=' ', header=None)
        label = float(file_paths[i][-7:-4])
        org = df_orig[sensor]
        ax.plot(range(len(org)), org, label = 'Original')
        
        ax.set_xlabel('Sensor ' + str(m[sensor]))
        sensor += 1

    plt.legend()
    plt.title(f'Counterfactual input for original input: +-{label}')
    plt.show()

    
    
# %%