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

from BNN.BNN import BayesianNeuralNetwork

#%% import files
folder_path = f'data/{DATASET}/min-max/test'  # Specify the path to your folder
head = ['Sensor 2', 'Sensor 3', 'Sensor 4', 'Sensor 7', 'Sensor 8', 'Sensor 9', 'Sensor 11', 'Sensor 12', 'Sensor 13', 'Sensor 14', 'Sensor 15', 'Sensor 17', 'Sensor 20', 'Sensor 21']

with open(os.path.join(folder_path, '0-Number_of_samples.csv')) as csvfile:
    sample_len = list(csv.reader(csvfile)) #list containing the amount of samples per engine/trajectory

file_paths = glob.glob(os.path.join(folder_path, '*.txt'))  # Get a list of all file paths in the folder
file_paths.sort() 

# Model input parameters
input_size = 14 #number of features
hidden_size = 32
num_layers = 1

#Import into trained machine learning models
BNNmodel = BayesianNeuralNetwork(input_size, hidden_size).to(device)
with open(f'BNN/BNN_model_state_{DATASET}_test.pt', 'rb') as f: 
    BNNmodel.load_state_dict(load(f)) 

for file_path in file_paths[0:1]:

    sample = np.genfromtxt(file_path, delimiter=" ", dtype=np.float32)
    label = float(file_path[-7:-4])

    df = pd.DataFrame(sample, columns=head)

    data = dice_ml.Data(datframe=df, continueous_features=df)

    model = dice_ml.Model(model=BNNmodel, backend='PYT')
    
# %%
