#%% Evaluating overall performance of models
import os
import glob
import csv
import numpy as np
import pandas as pd

import plotly.offline as pyo
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
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
        if key in pred_split_dict:
            alphas = []
            for pred in pred_split_dict[key][0]:
                mean, var, true = pred
                upper = plot_mean_and_percentile(mean, var, upper_lower='upper')
                lower = plot_mean_and_percentile(mean, var, upper_lower='lower')
                alpha = max(np.abs(true - upper), np.abs(true - lower))/true if int(true) != 0 else 0
                alphas.append(np.round(alpha,2))
            alpha_split_dict[str(key)] = max(alphas)
        else:
            alpha_split_dict[str(key)] = np.NaN

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
        if key in error_split_dict:
            flat_errors = list(chain(*error_split_dict[key]))
            squared_errors = [error**2 for error in flat_errors]
            RMSE = np.sqrt(np.mean(squared_errors))
            RMSE_splits[str(key)] = np.round(RMSE,2)
        else:
            RMSE_splits[str(key)] = np.NaN

    return dict(RMSE_splits)

def var_split(vars, key_ranges):
    # Create a dict for the variances
    var_dict = defaultdict(list)
    for key, value in zip(trues, vars):
        var_dict[key].append(value)

    # Convert the defaultdict to a regular dictionary
    var_dict = dict(var_dict)

    # Create a dict that splits the variance values over certain RUL sections
    var_split_dict = defaultdict(list)


    # Split up the variance according to their true RUL
    for key, value in var_dict.items():
        for high, low in key_ranges:
            if low <= key < high:
                var_split_dict[high, low].append(value)

    #take the average
    var_splits = defaultdict(list)
    for key in key_ranges:
        if key in var_split_dict:
            var_splits[str(key)] = np.average(var_split_dict[key])
        else:
            var_splits[str(key)] = np.NaN

    return dict(var_splits)

    
NOISY = False

if NOISY:
    test_path = f'{DATASET}/noisy'
else:
    test_path = f'{DATASET}'
    engine_eval = 0
#%%
#import BNN results: every file represents 1 engine
BNN_result_path = os.path.join(project_path, 'BNN/BNN_results', test_path)
engines= glob.glob(os.path.join(BNN_result_path, '*.json'))  # Get a list of all file paths in the folder
engines.sort() 

#import CF results: every file represents 1 engine
CF_result_path = os.path.join(project_path, 'DiCE_uncertainty/BNN_cf_results/outputs', test_path)
CF_engines= glob.glob(os.path.join(CF_result_path, '*.json'))  # Get a list of all file paths in the folder
CF_engines.sort() 

#import DNN results: every file represents 1 engine
DNN_result_path = os.path.join(project_path, 'BNN/DNN_results', test_path)
DNN_engines= glob.glob(os.path.join(DNN_result_path, '*.json'))  # Get a list of all file paths in the folder
DNN_engines.sort() 

#Gather data for all engines
BNN_scores = []
DNN_scores = []

B_means = [] #list of Bayesian prediction means
B_vars = [] #list of Bayesian prediction variance

D_preds = [] #list of deterministic predictions

CF_means = []
CF_vars = []

trues = [] #list of true RUl values

B_RMSE_lst = [] #list of Bayesian RMSE values
D_RMSE_lst = [] #list of Deterministic RMSE values
CF_RMSE_lst = [] #list of Counterfactual RMSE values

B_errors = [] #list of Bayesian errors
D_errors = [] #list of Deterministic errors
CF_errors = [] #list of Counterfactual errors

# engines = engines[engine_eval:engine_eval+1] if not TEST_SET else engines #only evaluate a single engine

for engine in engines:
    engine_id = int(engine[-8:-5])
    with open(engine, 'r') as jsonfile:
        results = json.load(jsonfile)
    
    #BNN results
    mean_pred_lst = results['mean'] #Mean of the RUL predictions over engine life
    var_pred_lst = results['var'] #Variance of the RUL predictions over engine life
    true_lst = results['true'] #Ground truth RUL over engine life

    with open(CF_engines[engine_id], 'r') as jsonfile:
        CF_results = json.load(jsonfile)
    
    #Counterfactual results
    CF_mean_pred_lst = CF_results['mean'] #Mean of the RUL predictions over engine life
    CF_var_pred_lst = CF_results['var'] #Variance of the RUL predictions over engine life
    CF_true_lst = CF_results['true'] #Ground truth RUL over engine life

    with open(DNN_engines[engine_id], 'r') as jsonfile:
        DNN_results = json.load(jsonfile)
    
    #Deterministic results
    y_pred_lst = DNN_results['pred'] #RUl predictions over engine life

        
    #%% Get error data
    BNN_error = [(mean_pred_lst[i] - true_lst[i]) for i in range(len(true_lst))] #BNN error
    DNN_error = [(y_pred_lst[i] - true_lst[i]) for i in range(len(true_lst))]#DNN error
    CF_error = [(CF_mean_pred_lst[i] - true_lst[i]) for i in range(len(true_lst))] #Counterfactual error
    B_RMSE = np.round(np.sqrt(np.mean([(mean_pred_lst[i] - true_lst[i])**2 for i in range(len(true_lst))])), 2) #Root Mean Squared error of Bayesian prediciton
    D_RMSE = np.round(np.sqrt(np.mean([(y_pred_lst[i] - true_lst[i])**2 for i in range(len(true_lst))] )), 2) #Root Mean Squared error of Deterministic prediciton
    CF_RMSE = np.round(np.sqrt(np.mean([(CF_mean_pred_lst[i] - true_lst[i])**2 for i in range(len(true_lst))] )), 2)

    B_RMSE_lst.append(B_RMSE)
    D_RMSE_lst.append(D_RMSE)
    CF_RMSE_lst.append(CF_RMSE)


    #Calculate scores
    BNN_score = sum([s_score(i) for i in BNN_error])
    DNN_score = sum([s_score(i) for i in DNN_error])

    BNN_scores.append(BNN_score)
    DNN_scores.append(DNN_score)

    B_means.append(mean_pred_lst)
    B_vars.append(var_pred_lst)
    D_preds.append(y_pred_lst)
    CF_means.append(CF_mean_pred_lst)
    CF_vars.append(CF_var_pred_lst)

    trues.append(true_lst)

    B_errors.append(BNN_error)
    D_errors.append(DNN_error)
    CF_errors.append(CF_error)


print(f'BNN score: {sum(BNN_scores)}')
print(f'DNN score: {sum(DNN_scores)}')



#flatten the lists
# means = list(chain(*means))
# vars = list(chain(*vars))
# det_preds = list(chain(*det_preds))
# trues = list(chain(*trues))
# B_errors = list(chain(*B_errors))
# D_errors = list(chain(*D_errors))

# #sort based on true RUL
# combined_lsts = list(zip(means, vars, det_preds, B_errors, D_errors, trues))
# sorted_lsts = sorted(combined_lsts, key=lambda x: x[-1], reverse=True)

# means, vars, det_preds, B_errors, D_errors, trues = zip(*sorted_lsts)

# Look at performance for different sections
# Define the key ranges for each category
key_ranges = [(float('inf'), 120), (120, 60), (60, 30), (30, 10), (10, 0)]

total_B_RMSE_splits = {str(key) : [] for key in key_ranges}
total_D_RMSE_splits = {str(key) : [] for key in key_ranges}
total_CF_RMSE_splits = {str(key) : [] for key in key_ranges}
total_B_var_splits = {str(key) : [] for key in key_ranges}
total_CF_var_splits = {str(key) : [] for key in key_ranges}
total_alpha_splits = {str(key) : [] for key in key_ranges}

for B_means, B_vars, B_errors, D_preds, D_errors, CF_means, CF_vars, CF_errors, trues in zip(B_means, B_vars, B_errors, D_preds, D_errors, CF_means, CF_vars, CF_errors, trues ):
    B_RMSE_splits = RMSE_split(B_errors, trues, key_ranges)
    D_RMSE_splits = RMSE_split(D_errors, trues, key_ranges)
    CF_RMSE_splits = RMSE_split(CF_errors, trues, key_ranges)

    B_var_splits = var_split(B_vars, key_ranges)
    CF_var_splits = var_split(CF_vars, key_ranges)

    B_alpha_splits = alpha_splits(B_means, B_vars, trues, key_ranges)

    for key in key_ranges:
        key = str(key)
        total_B_RMSE_splits[key].append(B_RMSE_splits[key])
        total_D_RMSE_splits[key].append(D_RMSE_splits[key])
        total_CF_RMSE_splits[key].append(CF_RMSE_splits[key])

        total_B_var_splits[key].append(B_var_splits[key])
        total_CF_var_splits[key].append(CF_var_splits[key])

        total_alpha_splits[key].append(B_alpha_splits[key])



ave_B_RMSE_splits = np.nanmean(list(total_B_RMSE_splits.values()), axis=1)
ave_D_RMSE_splits = np.nanmean(list(total_D_RMSE_splits.values()), axis=1)
ave_CF_RMSE_splits = np.nanmean(list(total_CF_RMSE_splits.values()), axis=1)

ave_B_var_splits = np.nanmean(list(total_B_var_splits.values()), axis=1)
ave_CF_var_splits = np.nanmean(list(total_CF_var_splits.values()), axis=1)

ave_alpha_splits = np.nanmean(list(total_alpha_splits.values()), axis=1)

tab = PrettyTable(key_ranges)
tab.add_row(np.round(ave_B_RMSE_splits, 2))
tab.add_row(np.round(ave_D_RMSE_splits, 2))
tab.add_row(np.round(ave_CF_RMSE_splits,2), divider=True)
tab.add_row(np.round(ave_B_var_splits, 2))
tab.add_row(np.round(ave_CF_var_splits,2), divider=True)
tab.add_row(np.round(ave_alpha_splits, 2), divider=True)
tab.add_column('Metric', ['Average Bayesian RMSE', 
                          'Average Deterministic RMSE', 
                          'Average Counterfactual RMSE', 
                          'Average Bayesian Variance', 
                          'Average Counterfactual variance', 
                          '90% alpha value (Bayesian)'], align='r')
print('RMSE (cycles) for RUL sections')
print(tab)

#plot results
x_plot = np.arange(len(B_means))


fig = make_subplots(rows=3, cols=1, subplot_titles=['RMSE error per section', 'Varaince per section', '90% alpha bounds per section'])

# for i, key in enumerate(key_ranges):
#     x_values = [str(key)] * len(total_B_RMSE_splits[key])
#     x_values = [str(key)]

# fig.add_trace(trace = px.box(pd.DataFrame(total_B_RMSE_splits)))

fig.add_trace(trace=go.Box(y=pd.DataFrame(total_B_RMSE_splits), name=f'Bayesian RMSE', marker_color='blue', boxmean=True), row=1, col=1)
fig.add_trace(trace=go.Box(y=pd.DataFrame(total_D_RMSE_splits), name=f'Deterministic RMSE', marker_color='orange', boxmean=True), row=1, col=1)
fig.add_trace(trace=go.Box(y=pd.DataFrame(total_CF_RMSE_splits), name=f'Counterfactual RMSE', marker_color='green', boxmean=True), row=1, col=1)
fig.add_trace(trace=go.Box(y=pd.DataFrame(total_B_var_splits), name=f'Bayesian variance', marker_color='blue', boxmean=True), row=2, col=1)
fig.add_trace(trace=go.Box(y=pd.DataFrame(total_CF_var_splits), name=f'Counterfactual variance', marker_color='green', boxmean=True), row=2, col=1)
fig.add_trace(trace=go.Box(y=pd.DataFrame(total_alpha_splits), name=f'90% alpha bound (Bayesian)', marker_color='blue', boxmean=True), row=3, col=1)

fig.update_layout(
    boxmode='group'
)
fig.update_yaxes(title_text='RMSE (cycles)', row=1, col=1)
fig.update_yaxes(title_text='Variance', row=2, col=1)
fig.update_yaxes(title_text='Relative distance from true RUL [-]', row=3, col=1)
fig.show()

# Export the figure to an HTML file
pyo.plot(fig, filename='interactive-plots/boxplots.html', auto_open=False)