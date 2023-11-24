#%% Evaluating overall performance of models
import os
import glob
import csv
import numpy as np
import pandas as pd

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
from collections import defaultdict
from prettytable import PrettyTable

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
    
def alpha_splits(means, vars, trues, key_ranges):

    pred_dict = defaultdict(list)
    for key, value in zip(trues, zip(means, vars, trues)):
        pred_dict[key].append(value)

    pred_dict = dict(pred_dict)

    pred_split_dict = defaultdict(list)

    for key, value in pred_dict.items():
        for high, low in key_ranges:
            if low <= key < high:
                pred_split_dict[high, low].append(value)

    pred_split_dict = dict(pred_split_dict)

    alpha_split_dict = defaultdict(list)

    for key in key_ranges:
        alphas = []
        for pred in pred_split_dict[key][0]:
            mean, var, true = pred
            upper = plot_mean_and_percentile(mean, var, upper_lower='upper')
            lower = plot_mean_and_percentile(mean, var, upper_lower='lower')
            alpha = max(np.abs(true - upper), np.abs(true - lower))/true if int(true) != 0 else 0
            alphas.append(np.round(alpha,2))
        alpha_split_dict[key] = max(alphas)

    alpha_split_dict = dict(alpha_split_dict)

    return alpha_split_dict
    
def RMSE_split(errors, trues, key_ranges):

    # Create a dict for the errors
    error_dict = defaultdict(list)
    for key, value in zip(trues, errors):
        error_dict[key].append(value)

    # Convert the defaultdict to a regular dictionary
    error_dict = dict(error_dict)

    # Create a dict that splits the error values over certain RUL sections
    error_split_dict = defaultdict(list)


    # Split up the errors according to their true RUL
    for key, value in error_dict.items():
        for high, low in key_ranges:
            if low <= key < high:
                error_split_dict[high, low].append(value)

    # Convert to dict split up into sections
    error_split_dict = dict(error_split_dict)

    RMSE_splits = defaultdict(list)
    for key in key_ranges:
        flat_errors = list(chain(*error_split_dict[key]))
        squared_errors = [error**2 for error in flat_errors]
        RMSE = np.sqrt(np.mean(squared_errors))
        RMSE_splits[key] = np.round(RMSE,2)

    return dict(RMSE_splits)

    
TEST_SET = True

if TEST_SET:
    test_path = f'{DATASET}/test'
else:
    test_path = f'{DATASET}'
    engine_eval = 0
#%%
#import BNN results: every file represents 1 engine
BNN_result_path = os.path.join(project_path, 'BNN/BNN_results', test_path)
engines= glob.glob(os.path.join(BNN_result_path, '*.json'))  # Get a list of all file paths in the folder
engines.sort() 

#import CF results: every file represents 1 engine
CF_result_path = os.path.join(project_path, 'DiCE/BNN_cf_results/outputs', DATASET)
CF_engines= glob.glob(os.path.join(CF_result_path, '*.json'))  # Get a list of all file paths in the folder
CF_engines.sort() 

#import DNN results: every file represents 1 engine
DNN_result_path = os.path.join(project_path, 'BNN/DNN_results', test_path)
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

B_errors = [] #list of Bayesian errors
D_errors = [] #list of Deterministic errors

engines = engines[engine_eval:engine_eval+1] if not TEST_SET else engines #only evaluate a single engine

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

    B_errors.append(BNN_error)
    D_errors.append(DNN_error)


print(f'BNN score: {sum(BNN_scores)}')
print(f'DNN score: {sum(DNN_scores)}')



#flatten the lists
means = list(chain(*means))
vars = list(chain(*vars))
det_preds = list(chain(*det_preds))
trues = list(chain(*trues))
B_errors = list(chain(*B_errors))
D_errors = list(chain(*D_errors))

#sort based on true RUL
combined_lsts = list(zip(means, vars, det_preds, B_errors, D_errors, trues))
sorted_lsts = sorted(combined_lsts, key=lambda x: x[-1], reverse=True)

means, vars, det_preds, B_errors, D_errors, trues = zip(*sorted_lsts)

# Look at performance for different sections
# Define the key ranges for each category
key_ranges = [(float('inf'), 120), (120, 60), (60, 30), (30, 10), (10, 0)]
# key_ranges = [(float('inf'), 0)]
B_RMSE_splits = RMSE_split(B_errors, trues, key_ranges)
D_RMSE_splits = RMSE_split(D_errors, trues, key_ranges)

B_alpha_splits = alpha_splits(means, vars, trues, key_ranges)

tab = PrettyTable(key_ranges)
tab.add_row(list(B_RMSE_splits.values()))
tab.add_row(list(D_RMSE_splits.values()), divider=True)
tab.add_row(list(B_alpha_splits.values()), divider=True)
tab.add_column('Metric', ['Bayesian RMSE', 'Deterministic RMSE', '90% alpha value'], align='r')
print('RMSE (cycles) for RUL sections')
print(tab)

#plot results
x_plot = np.arange(len(means))

plt.plot(means)
plt.fill_between(x_plot, 
                np.array([plot_mean_and_percentile(means[i], vars[i], percentile=90, upper_lower='upper') for i in range(len(x_plot))]), 
                np.array([plot_mean_and_percentile(means[i], vars[i], percentile=90, upper_lower='lower') for i in range(len(x_plot))]), 
                alpha = 0.5)

plt.plot(det_preds)
plt.plot(trues)
plt.show()