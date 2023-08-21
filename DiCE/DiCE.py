#Script to extract counterfactual explanations using DiCE package
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
from torch import load
import matplotlib.pyplot as plt
import random

import dice_ml_custom as dice_ml
from dice_ml_custom import Dice

from custom_BNN import CustomBayesianNeuralNetwork


#%% import files
TRAINDATASET = f'data/{DATASET}/min-max/train'
TESTDATASET = f'data/{DATASET}/min-max/test'

with open(os.path.join(project_path, TESTDATASET, '0-Number_of_samples.csv')) as csvfile:
    sample_len = list(csv.reader(csvfile)) #list containing the amount of samples per engine/trajectory

file_paths = glob.glob(os.path.join(project_path, TESTDATASET, '*.txt'))  # Get a list of all file paths in the folder
file_paths.sort() 


#Import into trained machine learning models
BNNmodel = CustomBayesianNeuralNetwork().to(device)
with open(f'{project_path}/BNN/BNN_model_state_{DATASET}_test.pt', 'rb') as f: 
    BNNmodel.load_state_dict(load(f)) 

#set Counterfactual hyperparameters
cf_amount = 1

#%%Go over each sample
for file_path in file_paths[0:1]:

    #load sample with true RUL
    sample = np.genfromtxt(file_path, delimiter=" ", dtype=np.float32)
    label = float(file_path[-7:-4])

    #Create labels for sensors and RUL
    sensors = [2,3,4,7,8,9,11,12,13,14,15,17,20,21]
    head = [[f'Sensor {i,j}' for j in range(len(sample)) for i in sensors]]
    head[0].append('RUL')

    #Flatten sample and combine with RUL
    sample = [[element for row in sample for element in row]] #flatten time series sample into format [(sensor 1, timestep 0),...(sensor n, timestep w)]
    sample = np.column_stack((sample, label))

    #Convert to dataframe and distinguish continuous features
    df = pd.DataFrame(sample, columns=head[0])
    df_continuous_features = df.drop('RUL', axis=1).columns.tolist()

    #Data and model object for DiCE
    data = dice_ml.Data(dataframe=df, continuous_features=df_continuous_features, outcome_name='RUL')
    model = dice_ml.Model(model=BNNmodel, backend='PYT', model_type='regressor')
    exp_random = Dice(data, model, method='random')
    exp_genetic = Dice(data, model, method='genetic')


    #Generate counterfactual explanations
    cf = exp_random.generate_counterfactuals(df.drop('RUL', axis=1), total_CFs= cf_amount, desired_range=[label+5, label+20], proximity_weight= 0.02, random_seed = 2)
    cf.visualize_as_dataframe(show_only_changes=True)
    

    cf_df = cf.cf_examples_list[0].final_cfs_df
    cf_df = cf_df.drop('RUL', axis=1).values.reshape(30,14)
    cf_df = pd.DataFrame(cf_df)
    

#%% Plot counterfacutal dataframe
df_orig = pd.read_csv(f'{project_path}/data/FD001/min-max/test/test_00000-120.txt', sep=' ', header=None)

fig, axes = plt.subplots(nrows=2, ncols=7, sharex=True,
                                        figsize=(25, 8))

i = 0
m = [2,3,4,7,8,9,11,12,13,14,15,17,20,21]
for ax in axes.ravel():
    counter = cf_df[i]
    org = df_orig[i]
    ax.plot(range(len(org)), org, label = 'Original')
    ax.plot(range(len(counter)), counter, label='Counterfactual', linestyle='--')
    ax.set_xlabel('Sensor ' + str(m[i]))
    i += 1

plt.legend()
plt.show()
    
# %%