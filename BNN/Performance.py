#%% Evaluating overall performance of models
import os
import glob
import csv
import numpy as np

import plotly.offline as pyo
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import norm

import matplotlib.pyplot as plt

from torch import load
from tqdm import tqdm
import torch
from torchvision.transforms import ToTensor
import time
import sys
import scipy.stats as stats
import json
from itertools import chain

# Get the absolute path of the project directory
project_path = os.path.dirname(os.path.abspath(os.path.join((__file__), os.pardir)))
# Add the project directory to sys.path if it's not already present
if project_path not in sys.path:
    sys.path.append(project_path)

from DNN import NeuralNetwork
# from DNN_vizualizer import y_pred_lst, D_RMSE

from variables import *

def s_score(error):
    """S score specifically for RUL prediction. The characteristic of this scoring funciton is that it favours early predictions
    more than late predicitons.

    Args:
        error (float): Predicted RUL - True RUL

    Returns:
        float: scoring function result
    """
    if error < 0:
        return np.exp(-error/13) - 1
    else:
        return np.exp(error/10) - 1
    
def plot_mean_and_percentile(mean, variance, percentile=90, upper_lower = 'upper'):
    std_dev = np.sqrt(variance)

    # Calculate the z-score corresponding to the desired percentile
    z_score = stats.norm.ppf(percentile / 100)

    # Calculate the values corresponding to the mean, lower percentile, and upper percentile
    lower_percentile_value = mean - z_score * std_dev
    upper_percentile_value = mean + z_score * std_dev

    if upper_lower == 'upper':
        return upper_percentile_value
    elif upper_lower == 'lower':
        return lower_percentile_value

engine_eval = 0
#%%
#import BNN results: every file represents 1 engine
BNN_result_path = os.path.join(project_path, 'BNN/BNN_results', DATASET, 'test')
engines= glob.glob(os.path.join(BNN_result_path, '*.json'))  # Get a list of all file paths in the folder
engines.sort() 

#import CF results: every file represents 1 engine
CF_result_path = os.path.join(project_path, 'DiCE/BNN_cf_results/outputs', DATASET)
CF_engines= glob.glob(os.path.join(CF_result_path, '*.json'))  # Get a list of all file paths in the folder
CF_engines.sort() 

#import DNN results: every file represents 1 engine
DNN_result_path = os.path.join(project_path, 'BNN/DNN_results', DATASET, 'test')
DNN_engines= glob.glob(os.path.join(DNN_result_path, '*.json'))  # Get a list of all file paths in the folder
DNN_engines.sort() 

#Gather data for all engines
BNN_scores = []
DNN_scores = []

means = [] #list of prediction means
vars = [] #list of prediction variance

det_preds = [] #list of deterministic predictions

trues = [] #list of true RUl values

B_RMSE_lst = [] #list of Bayesian RMSE values
D_RMSE_lst = [] #list of Deterministic RMSE values

for engine in engines:
    engine_id = int(engine[-8:-5])
    with open(engine, 'r') as jsonfile:
        results = json.load(jsonfile)
    
    #BNN results
    mean_pred_lst = results['mean'] #Mean of the RUL predictions over engine life
    var_pred_lst = results['var'] #Variance of the RUL predictions over engine life
    true_lst = results['true'] #Ground truth RUL over engine life

    # with open(CF_engines[engine_id], 'r') as jsonfile:
    #     CF_results = json.load(jsonfile)
    
    # #Counterfactual results
    # CF_mean_pred_lst = CF_results['mean'] #Mean of the RUL predictions over engine life
    # CF_var_pred_lst = CF_results['var'] #Variance of the RUL predictions over engine life
    # CF_true_lst = CF_results['true'] #Ground truth RUL over engine life

    with open(DNN_engines[engine_id], 'r') as jsonfile:
        DNN_results = json.load(jsonfile)
    
    #Deterministic results
    y_pred_lst = DNN_results['pred'] #RUl predictions over engine life

        
    #%% Get error data
    BNN_error = [(mean_pred_lst[i] - true_lst[i]) for i in range(len(true_lst))] #BNN error
    DNN_error = [(y_pred_lst[i] - true_lst[i]) for i in range(len(true_lst))]#DNN error
    B_RMSE = np.round(np.sqrt(np.mean([(mean_pred_lst[i] - true_lst[i])**2 for i in range(len(true_lst))])), 2) #Root Mean Squared error of Bayesian prediciton
    D_RMSE = np.round(np.sqrt(np.mean([(y_pred_lst[i] - true_lst[i])**2 for i in range(len(true_lst))] )), 2) #Root Mean Squared error of Deterministic prediciton

    B_RMSE_lst.append(B_RMSE)
    D_RMSE_lst.append(D_RMSE)


    #Calculate scores
    BNN_score = sum([s_score(i) for i in BNN_error])
    DNN_score = sum([s_score(i) for i in DNN_error])

    BNN_scores.append(BNN_score)
    DNN_scores.append(DNN_score)

    means.append(mean_pred_lst)
    vars.append(var_pred_lst)
    det_preds.append(y_pred_lst)

    trues.append(true_lst)


print(f'BNN score: {sum(BNN_scores)} - RMSE: {np.mean(B_RMSE_lst)}')
print(f'DNN score: {sum(DNN_scores)} - RMSE: {np.mean(D_RMSE_lst)}')



#flatten the lists
means = list(chain(*means))
vars = list(chain(*vars))
det_preds = list(chain(*det_preds))
trues = list(chain(*trues))

#sort based on true RUL
combined_lsts = list(zip(means, vars, det_preds, trues))
sorted_lsts = sorted(combined_lsts, key=lambda x: x[-1], reverse=True)

means, vars, det_preds, trues = zip(*sorted_lsts)
x_plot = np.arange(len(means))

plt.plot(means)
plt.fill_between(x_plot, 
                np.array([plot_mean_and_percentile(means[i], vars[i], percentile=90, upper_lower='upper') for i in range(len(x_plot))]), 
                np.array([plot_mean_and_percentile(means[i], vars[i], percentile=90, upper_lower='lower') for i in range(len(x_plot))]), 
                alpha = 0.5)

plt.plot(det_preds)
plt.plot(trues)
plt.show()