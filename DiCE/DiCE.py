#Script to extract counterfactual explanations using DiCE package
#%% import dependencies
import sys
import os
import shutil
import importlib

# Get the absolute path of the project directory
project_path = os.path.dirname(os.path.abspath(os.path.join((__file__), os.pardir)))
# Add the project directory to sys.path if it's not already present
if project_path not in sys.path:
    sys.path.append(project_path)
if os.path.join(project_path, "DiCE/BNN_copies") not in sys.path:
    sys.path.append(os.path.join(project_path, "DiCE/BNN_copies"))

import glob
import csv
import numpy as np
import pandas as pd
from torch import load
import multiprocessing as mp
import time
from p_tqdm import p_map
import tqdm

import warnings
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

# import dice_ml_custom as dice_ml_custom


device = 'cpu' #device where models whill be run
DATASET = 'FD001' #which data set to use from cmpass [FD001, FD002, FD003, FD004]

BATCHSIZE = 100
EPOCHS = 100

k = 10 #amount of folds for cross validation



# def load_required_packages():
#     import dice_ml_custom as dice_ml_custom
#     from dice_ml_custom import Dice

#     import numpy as np
#     import pandas as pd

#     import warnings
#     warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

# load_required_packages()

#%% import files
TRAINDATASET = f'data/{DATASET}/min-max/train'
TESTDATASET = f'data/{DATASET}/min-max/test'

BayDet = 'BNN'

with open(os.path.join(project_path, TESTDATASET, '0-Number_of_samples.csv')) as csvfile:
    sample_len = list(csv.reader(csvfile)) #list containing the amount of samples per engine/trajectory



#set Counterfactual hyperparameters
cf_amount = 1

#%%Go over each sample
def CMAPSS_counterfactuals(file_path):
    current_process = int(mp.current_process().name[15:])

    custom_BNN = importlib.import_module(f'BNN_copy_{current_process}')
    
    CustomBayesianNeuralNetwork = custom_BNN.CustomBayesianNeuralNetwork

    #Import into trained machine learning models
    if BayDet == 'BNN':
        model = CustomBayesianNeuralNetwork()
    
    
    with open(f'{project_path}/BNN/model_states/{BayDet}_model_state_{DATASET}_test.pt', 'rb') as f: 
        model.load_state_dict(load(f)) 

    model.eval()

    
    dice_ml_custom = importlib.import_module('dice_ml_custom', os.path.join(project_path, f'dice_copies/dice_copy_{current_process}/dice_ml_custom'))
    

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
    data = dice_ml_custom.Data(dataframe=df, continuous_features=df_continuous_features, outcome_name='RUL')
    dice_model = dice_ml_custom.Model(model=model, backend='PYT', model_type='regressor')
    exp_random = dice_ml_custom.Dice(data, dice_model, method='random')



    #Generate counterfactual explanations
    cf_amount = 1
    print(f'generating counterfactual {current_process}')
    cf = exp_random.generate_counterfactuals(df.drop('RUL', axis=1), 
                                             verbose=False, 
                                             total_CFs= cf_amount, 
                                             desired_range=[3, 6], 
                                             proximity_weight= 0.0002, 
                                             random_seed = 2, 
                                             time_series=True)
    
    # cf.visualize_as_dataframe(show_only_changes=True)
    
    cf_total = cf.cf_examples_list[0].final_cfs_df
    print(f'Counterfactual {current_process} processed')
    
    if cf_total is not None:
        #Save cf_result to file
        save_to = os.path.join(project_path, f'DiCE/{BayDet}_cf_results/inputs', DATASET)
        if not os.path.exists(save_to): os.makedirs(save_to)
        file_name = os.path.join(save_to, "cf_{0:0=5d}_{1:0=3d}.csv".format(sample_id, int(cf_total['RUL'])))
        cf_total.to_csv(file_name, index=False)
        # print(f'Saved to: {file_name}')

    else:
        #If no cf found, save a file containing NaN
        save_to = os.path.join(project_path, f'DiCE/{BayDet}_cf_results/inputs', DATASET)
        if not os.path.exists(save_to): os.makedirs(save_to)
        file_name = os.path.join(save_to, "cf_{0:0=5d}_{1:0=3d}.csv".format(sample_id, label))
        no_cf = pd.DataFrame([[np.NAN for _ in range(len(sample[0]))]], columns=head[0])
        no_cf.to_csv(file_name, index=False)
        # print(f'Saved to: {file_name}')


if __name__ == '__main__':

    start = time.time()

    num_cores = mp.cpu_count()

    #make copies of dice package folder to prevent GIL
    source_folder = os.path.join(project_path, 'dice_ml_custom')
    num_copies = num_cores
    copy_folder = os.path.join(project_path, 'dice_copies')
    if not os.path.exists(copy_folder): 
        os.makedirs(copy_folder)
        for i in range(num_copies):
            copy_name = os.path.join(copy_folder, f'dice_copy_{i+1}/dice_ml_custom')
            shutil.copytree(source_folder, copy_name)

    #make copies of models to prevent GIL
    source_folder = os.path.join(project_path, 'DiCE/custom_BNN.py')
    num_copies = num_cores
    copy_folder = os.path.join(project_path, 'DiCE/BNN_copies')
    if not os.path.exists(copy_folder): 
        os.makedirs(copy_folder)
        for i in range(num_copies):
            copy_name = os.path.join(copy_folder, f'BNN_copy_{i+1}.py')
            shutil.copy(source_folder, copy_name)



    file_paths = glob.glob(os.path.join(project_path, TESTDATASET, '*.txt'))  # Get a list of all file paths in the folder
    file_paths.sort()
    file_paths = file_paths[0:int(sample_len[0][0])] #only looking at the first engine
    print('Starting multiprocessing')
    print(f'Number of available cores: {num_cores}')
    print(f'Number of samples: {len(file_paths)}')


    with mp.Pool(processes=num_cores) as pool:
        list(tqdm.tqdm(pool.imap_unordered(CMAPSS_counterfactuals, file_paths), total=len(file_paths)))

    # p_map(CMAPSS_counterfactuals, file_paths, num_cpus=num_cores, total=len(file_paths), desc= 'Processing')

    end = time.time()
    print('Processing ended')
    print('Time elapsed:', np.round((end-start)/60, 2), 'minutes')



    #%%