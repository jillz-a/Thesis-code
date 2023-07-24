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

import dice_ml
from dice_ml import Dice

from custom_BNN import CustomBayesianNeuralNetwork

#%% import files
TRAINDATASET = os.path.abspath(f'data/{DATASET}/min-max/train')
TESTDATASET = os.path.abspath(f'data/{DATASET}/min-max/test')

with open(os.path.join(TESTDATASET, '0-Number_of_samples.csv')) as csvfile:
    sample_len = list(csv.reader(csvfile)) #list containing the amount of samples per engine/trajectory

file_paths = glob.glob(os.path.join(TESTDATASET, '*.txt'))  # Get a list of all file paths in the folder
file_paths.sort() 


#Import into trained machine learning models
BNNmodel = CustomBayesianNeuralNetwork().to(device)
with open(f'BNN/BNN_model_state_{DATASET}_test.pt', 'rb') as f: 
    BNNmodel.load_state_dict(load(f)) 

#set Counterfactual hyperparameters
cf_amount = 5

#Go over each sample
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

    #Generate counterfactual explanations
    cf = exp_random.generate_counterfactuals(df.drop('RUL', axis=1), total_CFs= cf_amount, desired_range=[123, 132])
    cf.visualize_as_dataframe(show_only_changes=True)
    #master branch
    
    
    cf_df = cf.cf_examples_list[0].final_cfs_df
    
    
# %%
