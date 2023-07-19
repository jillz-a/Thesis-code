#%%Vizualizing script for the ML models
import os
import glob
import csv
import numpy as np
import matplotlib.pyplot as plt
from torch import nn, load
from tqdm import tqdm
import torch
from torchvision.transforms import ToTensor
import time

from DNN import NeuralNetwork
from BNN import BayesianNeuralNetwork
from DNN_vizualizer import y_pred_lst, D_RMSE

from variables import *

start = time.time()

folder_path = f'data/{DATASET}/min-max/test'  # Specify the path to your folder

with open(os.path.join(folder_path, '0-Number_of_samples.csv')) as csvfile:
    sample_len = list(csv.reader(csvfile)) #list containing the amount of samples per engine/trajectory

file_paths = glob.glob(os.path.join(folder_path, '*.txt'))  # Get a list of all file paths in the folder
file_paths.sort() 

engines = [10]
for engine in engines:
    index = sum([int(sample_len[0:i+1][i][0]) for i in range(engine)])
    selected_file_paths = file_paths[index:index + int(sample_len[engine][0])]  # Select the desired number of files

    #setup data to plot
    mean_pred_lst = []
    true_lst = []
    var_pred_lst = []

    # Model input parameters
    input_size = 14 #number of features
    hidden_size = 32
    num_layers = 1

    #%%Go through each sample
    loop = tqdm(selected_file_paths)
    for file_path in loop:
    
        # Process each selected file
        sample = np.genfromtxt(file_path, delimiter=" ", dtype=np.float32)
        label = float(file_path[-7:-4])

        #Import into trained machine learning models
        NNmodel = BayesianNeuralNetwork(input_size, hidden_size).to(device)
        with open(f'BNN/BNN_model_state_{DATASET}_test.pt', 'rb') as f: 
            NNmodel.load_state_dict(load(f)) 

        #predict RUL from samples using Monte Carlo Sampling
        X = ToTensor()(sample).to(device)
        n_samples = 20

        mc_pred = [NNmodel(X)[0] for _ in range(n_samples)]


        predictions = torch.stack(mc_pred)
        mean_pred = torch.mean(predictions, dim=0)
        var_pred = torch.var(predictions, dim=0)
        y = label #True RUL

        #add predictions and true labels to lists
        mean_pred_lst.append(mean_pred.item())
        true_lst.append(y)
        var_pred_lst.append(var_pred.item())
        
        loop.set_description(f"Processing engine {engine}")


    error = [(mean_pred_lst[i] - true_lst[i])**2 for i in range(len(true_lst))]
    B_RMSE = np.round(np.sqrt(np.mean(error)), 2)

    plt.plot(mean_pred_lst, label= f'Bayesian Mean Predicted RUL values for engine {engine}, RMSE = {B_RMSE}')
    # plt.plot(y_pred_lst, label=f'Deterministic Predicted RUL values, RMSE = {D_RMSE}')
    plt.plot(true_lst, label='True RUL values')
    plt.fill_between(x=np.arange(len(mean_pred_lst)), 
                     y1= mean_pred_lst + np.sqrt(var_pred_lst), 
                     y2=mean_pred_lst - np.sqrt(var_pred_lst),
                     alpha= 0.5,
                    #  color = 'yellow',
                     label= '1 STD interval'
                     )
    plt.fill_between(x=np.arange(len(mean_pred_lst)), 
                     y1= mean_pred_lst + 2*np.sqrt(var_pred_lst), 
                     y2=mean_pred_lst - 2*np.sqrt(var_pred_lst),
                     alpha= 0.3,
                    #  color = 'yellow',
                     label= '2 STD interval'
                     )
#%%

finish = time.time()
print(f'elapsed time = {finish - start} seconds')
plt.xlabel('Cycles')
plt.ylabel('RUL')
plt.grid()
plt.title(f'Dataset {DATASET}, {n_samples} samples per data point, average variance = {np.round(np.mean(var_pred_lst),2)}')
plt.legend()
plt.show()