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

#%% import files
TRAINDATASET = f'data/{DATASET}/min-max/train'
TESTDATASET = f'data/{DATASET}/min-max/test'

result_path = os.path.join(project_path, 'DiCE/results', DATASET)
samples = glob.glob(os.path.join(result_path, '*.csv'))  # Get a list of all file paths in the folder

file_paths = glob.glob(os.path.join(project_path, TESTDATASET, '*.txt'))  # Get a list of all file paths in the folder
file_paths.sort() 

#%% Plot counterfacutal dataframe
for i, sample in enumerate(samples):
    cf_total = pd.read_csv(sample)
    cf_RUL = cf_total['RUL']
    cf_total = cf_total.drop('RUL', axis=1)

    df_orig = pd.read_csv(f'{project_path}/data/FD001/min-max/test/test_00000-120.txt', sep=' ', header=None)
    df_orig = pd.read_csv(file_paths[i], sep=' ', header=None)
    label = float(file_paths[i][-7:-4])

    fig, axes = plt.subplots(nrows=2, ncols=7, sharex=True,
                                            figsize=(25, 8))

    sensor = 0
    m = [2,3,4,7,8,9,11,12,13,14,15,17,20,21]
    for ax in axes.ravel():

        for i in range(len(cf_total)):
            cf_df = cf_total.iloc[[i]]
            cf_df = cf_df.values.reshape(30,14)
            cf_df = pd.DataFrame(cf_df)

            counter = cf_df[sensor]
            ax.plot(range(len(counter)), counter, label=f'CF {i + 1}: RUL = {cf_RUL.iloc[i]}', linestyle='--')

        org = df_orig[sensor]
        ax.plot(range(len(org)), org, label = 'Original')
        
        ax.set_xlabel('Sensor ' + str(m[sensor]))
        sensor += 1

    plt.legend()
    plt.title(f'Counterfactual input for original input: +-{label}')
    plt.show()
    
# %%