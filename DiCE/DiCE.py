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
import multiprocessing as mp
import time
import tqdm

import dice_ml_custom as dice_ml
from dice_ml_custom import Dice

from custom_BNN import CustomBayesianNeuralNetwork

import warnings
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)


#%% import files
TRAINDATASET = f'data/{DATASET}/min-max/train'
TESTDATASET = f'data/{DATASET}/min-max/test'

with open(os.path.join(project_path, TESTDATASET, '0-Number_of_samples.csv')) as csvfile:
    sample_len = list(csv.reader(csvfile)) #list containing the amount of samples per engine/trajectory


#Import into trained machine learning models
BNNmodel = CustomBayesianNeuralNetwork().to(device)
with open(f'{project_path}/BNN/BNN_model_state_{DATASET}_test.pt', 'rb') as f: 
    BNNmodel.load_state_dict(load(f)) 

#set Counterfactual hyperparameters
cf_amount = 1
#%%Go over each sample
def CMAPSS_counterfactuals(file_path):
    
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
    sample = np.column_stack((sample, label))

    #Convert to dataframe and distinguish continuous features
    df = pd.DataFrame(sample, columns=head[0])
    df_continuous_features = df.drop('RUL', axis=1).columns.tolist()

    #Data and model object for DiCE
    data = dice_ml.Data(dataframe=df, continuous_features=df_continuous_features, outcome_name='RUL')
    model = dice_ml.Model(model=BNNmodel, backend='PYT', model_type='regressor')
    exp_random = Dice(data, model, method='random')



    #Generate counterfactual explanations
    cf = exp_random.generate_counterfactuals(df.drop('RUL', axis=1), 
                                             verbose=False, 
                                             total_CFs= cf_amount, 
                                             desired_range=[3, 6], 
                                             proximity_weight= 0.0002, 
                                             random_seed = 2, 
                                             time_series=True)
    
    # cf.visualize_as_dataframe(show_only_changes=True)
    
    cf_total = cf.cf_examples_list[0].final_cfs_df
    
   
    #Save cf_result to file
    save_to = os.path.join(project_path, 'DiCE/results', DATASET)
    if not os.path.exists(save_to): os.makedirs(save_to)
    file_name = os.path.join(save_to, "cf_{0:0=5d}_{1:0=3d}.csv".format(sample_id, int(label)))
    cf_total.to_csv(file_name, index=False)


if __name__ == '__main__':

    start = time.time()
    print('Starting multiprocessing')
    file_paths = glob.glob(os.path.join(project_path, TESTDATASET, '*.txt'))  # Get a list of all file paths in the folder
    file_paths.sort()

    file_paths = file_paths[0:10]

    num_cores = mp.cpu_count()

    with mp.Pool(processes=num_cores) as pool:

        pool.map(CMAPSS_counterfactuals, file_paths)

    end = time.time()
    print('Processing ended')
    print('Time elapsed:', end-start, 'seconds')