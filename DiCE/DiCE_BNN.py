#%% import dependencies
import glob
import sys
import os
import csv
import json

os.environ["MKL_NUM_THREADS"] = "1" 
os.environ["NUMEXPR_NUM_THREADS"] = "1" 
os.environ["OMP_NUM_THREADS"] = "1" 
os.environ['OPENBLAS_NUM_THREADS'] = '1'

# Get the absolute path of the project directory
project_path = os.path.dirname(os.path.abspath(os.path.join((__file__), os.pardir)))
# Add the project directory to sys.path if it's not already present
if project_path not in sys.path:
    sys.path.append(project_path)

import numpy as np
import torch
import tqdm
from sklearn.model_selection import KFold
import pandas as pd

from torch import nn, save, load
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import multiprocessing as mp

import bayesian_torch.layers as bl

from variables import *
torch.manual_seed(42)

CF_DATASET = os.path.abspath(os.path.join(project_path, f'DiCE/BNN_cf_results/inputs/{DATASET}'))
folder_path = f'data/{DATASET}/min-max/test'  # Specify the path to your input folder
with open(os.path.join(project_path, folder_path, '0-Number_of_samples.csv')) as csvfile:
        sample_len = list(csv.reader(csvfile)) #list containing the amount of samples per engine/trajectory

SAVE = True #if true, result will be saved to json files

#Bayesian neural network class
class BayesianNeuralNetwork(nn.Module):
    """Bayesian Neural Network using LSTM and linear layers. Deterministic to Bayesian using Reparameterization.

    Args:
        input_size: number of input features
        hidden_szie: size of hidden node vector (also size of output)
        num_layers: amountof LSTM layers
        prior_mean: initial guess for parameter mean
        prior_variance: initial guess for parameter variance
        posterior_mu_init: init std for the trainable mu parameter, sampled from N(0, posterior_mu_init)
        posterior_rho_init: init std for the trainable rho parameter, sampled from N(0, posterior_rho_init)

    """
    def __init__(self, input_size=14, hidden_size=32, num_layers=1, prior_mean = 0.0, prior_variance = 1.0, posterior_mu_init = 0.0, posterior_rho_init = -3.0):
        super(BayesianNeuralNetwork, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = bl.LSTMReparameterization(in_features= input_size, out_features= hidden_size, prior_mean=prior_mean, prior_variance=prior_variance, posterior_mu_init=posterior_mu_init, posterior_rho_init=posterior_rho_init)
        self.relu = bl.ReLU()
        self.l1 = bl.LinearReparameterization(in_features=hidden_size, out_features=16)
        self.l2 = bl.LinearReparameterization(16,1)
        
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device) #initial hidden state
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device) #initial cell state
        
        out = self.lstm(x)#, (h0, c0))
        
        out = out[0][:, -1, :]  # Extract the last time step output
        
        out = self.l1(out) #pass through dense layers
       
        out = self.l2(out[0])
    
        return out

def open_cf(file_path):
    cf_data = []
    with open(file_path, 'r') as file:
        csv_reader = csv.reader(file)
        for row_number, row in enumerate(csv_reader):
            if row_number == 1:  # Second row (0-based index)
                cf_data = [np.float32(value) for value in row]

    # Step 2: Remove the final entry
    cf_RUL = cf_data[-1]
    cf_data = cf_data[:-1]

    # Step 3: Convert the modified second row into a 2D NumPy array
    shape = (30, 14)  # Desired shape
    array = np.array(cf_data).reshape(shape)

    return array, cf_RUL

# Function to split a list into chunks
def chunk_list(input_list, num_chunks):
    avg_chunk_size = len(input_list) // num_chunks
    remainder = len(input_list) % num_chunks

    chunks = []
    start = 0

    for i in range(num_chunks):
        chunk_size = avg_chunk_size + (1 if i < remainder else 0)
        end = start + chunk_size
        chunks.append(input_list[start:end])
        start = end

    return chunks

def CF_results(chunk):
    file_paths = glob.glob(os.path.join(CF_DATASET, '*.csv'))  # Get a list of all counterfactual input files
    file_paths.sort() 

    engines = chunk

    # Model input parameters
    input_size = 14 #number of features
    hidden_size = 32
    num_layers = 1

    #Import into trained machine learning models
    NNmodel = BayesianNeuralNetwork(input_size, hidden_size).to(device)
    with open(f'{project_path}/BNN/model_states/BNN_model_state_{DATASET}_test.pt', 'rb') as f: 
        NNmodel.load_state_dict(load(f)) 

    for engine in engines:
        index = sum([int(sample_len[0:i+1][i][0]) for i in range(engine)])
        selected_file_paths = file_paths[index:index + int(sample_len[engine][0])]  # Select the desired number of files
        # selected_file_paths = file_paths[0:170]

        #setup data to plot
        mean_pred_lst = []
        true_lst = []
        var_pred_lst = []


        #Go through each sample
        for file_path in selected_file_paths:
        
            # Process each selected file
            sample, cf_RUL = open_cf(file_path)
            label = float(file_path[-7:-4])


            #predict RUL from samples using Monte Carlo Sampling
            X = ToTensor()(sample).to(device)
            n_samples = 10

            mc_pred = [NNmodel(X)[0] for _ in range(n_samples)]


            predictions = torch.stack(mc_pred)
            mean_pred = torch.mean(predictions, dim=0)
            # print(mean_pred, cf_RUL)
            var_pred = torch.var(predictions, dim=0)
            y = label #True RUL

            #add predictions and true labels to lists
            mean_pred_lst.append(mean_pred.item())
            true_lst.append(y)
            var_pred_lst.append(var_pred.item())

        error = [(mean_pred_lst[i] - true_lst[i])**2 for i in range(len(true_lst))] #squared BNN error
        B_RMSE = np.round(np.sqrt(np.mean(error)), 2) #Root Mean Squared error of Bayesian prediciton

        #save engine results to file
        if SAVE:
            results = {
                'mean': mean_pred_lst,
                'var': var_pred_lst,
                'true': true_lst,
                'RMSE': B_RMSE
            }

            save_to = os.path.join(project_path, 'DiCE/BNN_cf_results/outputs', DATASET)
            if not os.path.exists(save_to): os.makedirs(save_to)
            file_name = os.path.join(save_to, "cf_result_{0:0=3d}.json".format(engine))
            
            with open(file_name, 'w') as jsonfile:
                json.dump(results, jsonfile)



if __name__ == "__main__":
    
    
    num_cores = mp.cpu_count()

    engines = np.arange(len(sample_len))
    # engines = [0]

    chunks = chunk_list(engines, min(num_cores, len(engines)))


    with mp.Pool(processes=min(num_cores, len(chunks))) as pool:
        list(tqdm.tqdm(pool.imap_unordered(CF_results, chunks), total=len(chunks)))