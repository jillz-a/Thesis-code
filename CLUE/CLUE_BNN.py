#Use the CLUE-master package to find counterfactuals with lower uncertainty

#%% import dependencies
import sys
import os
os.environ["MKL_NUM_THREADS"] = "1" 
os.environ["NUMEXPR_NUM_THREADS"] = "1" 
os.environ["OMP_NUM_THREADS"] = "1" 
os.environ['OPENBLAS_NUM_THREADS'] = '1'

# Get the absolute path of the project directory
project_path = os.path.dirname(os.path.abspath(os.path.join((__file__), os.pardir)))
# Add the project directory to sys.path if it's not already present
if project_path not in sys.path:
    sys.path.append(project_path)

import glob
import csv
import numpy as np
import pandas as pd
from torch import load
import multiprocessing as mp
import time
import tqdm

from DiCE.custom_BNN import CustomBayesianNeuralNetwork
from DiCE.custom_DNN import CustomNeuralNetwork

from CLUE_master.interpret import CLUE

from variables import *

#%% import files
TRAINDATASET = f'data/{DATASET}/min-max/train'
TESTDATASET = f'data/{DATASET}/min-max/test'

BayDet = 'BNN'

with open(os.path.join(project_path, TESTDATASET, '0-Number_of_samples.csv')) as csvfile:
    sample_len = list(csv.reader(csvfile)) #list containing the amount of samples per engine/trajectory

#Import into trained machine learning models
if BayDet == 'BNN':
    NNmodel = CustomBayesianNeuralNetwork().to(device)
elif BayDet == 'DNN':
    model = CustomNeuralNetwork().to(device)

with open(f'{project_path}/BNN/model_states/{BayDet}_model_state_{DATASET}_test.pt', 'rb') as f: 
    NNmodel.load_state_dict(load(f)) 

#Import input file paths
file_paths = glob.glob(os.path.join(project_path, TESTDATASET, '*.txt'))  # Get a list of all file paths in the folder
file_paths.sort()
file_path = file_paths[0]

#load sample with true RUL
sample = np.genfromtxt(file_path, delimiter=" ", dtype=np.float32)
sample_id = int(file_path[-13:-8])
label = int(file_path[-7:-4])

#Create labels for sensors and RUL
sensors = [2,3,4,7,8,9,11,12,13,14,15,17,20,21]
head = [[f'Sensor {i,j}' for j in range(len(sample)) for i in sensors]]
head[0].append('RUL')

#Flatten sample and combine with RUL
sample = [[element for row in sample for element in row]] #flatten time series sample into format [(sensor 1, timestep 0),...(sensor n, timestep w)]
# sample = np.column_stack((sample, label))

