#%%Vizualizing script for the BNN model including DNN and uncertainty predictions
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
import seaborn as sns

from tqdm import tqdm
import time
import sys
import scipy.stats as stats
import json

# Get the absolute path of the project directory
project_path = os.path.dirname(os.path.abspath(os.path.join((__file__), os.pardir)))
# Add the project directory to sys.path if it's not already present
if project_path not in sys.path:
    sys.path.append(project_path)

from DNN import NeuralNetwork
# from DNN_vizualizer import y_pred_lst, D_RMSE

from variables import *

#definitions
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

engine_eval = 3

NOISY = False
CF = False
NOCF = False
INCREASE = True

noisy = 'noisy' if NOISY else 'denoised'
cf = 'CF' if CF else ('NOCF' if NOCF else 'orig')
increase = 'increase' if INCREASE else 'decrease'
#%%
#import BNN results: every file represents 1 engine
BNN_result_path = os.path.join(project_path, 'BNN/BNN_results', DATASET, f'{noisy}-{cf}')
BNN_engines= glob.glob(os.path.join(BNN_result_path, '*.json'))  # Get a list of all file paths in the folder
BNN_engines.sort() 

#import CF results: every file represents 1 engine
CF_result_path = os.path.join(project_path, 'DiCE/BNN_cf_results/outputs', DATASET, increase, noisy)
CF_engines= glob.glob(os.path.join(CF_result_path, '*.json'))  # Get a list of all file paths in the folder
CF_engines.sort() 

#import DNN results: every file represents 1 engine
DNN_result_path = os.path.join(project_path, 'BNN/DNN_results', DATASET, noisy)
DNN_engines= glob.glob(os.path.join(DNN_result_path, '*.json'))  # Get a list of all file paths in the folder
DNN_engines.sort() 

for BNN_engine in BNN_engines[engine_eval: engine_eval+1]:
    engine_id = int(BNN_engine[-8:-5])
    with open(BNN_engine, 'r') as jsonfile:
        BNN_results = json.load(jsonfile)
    
    #BNN results
    mean_pred_lst = BNN_results['mean'] #Mean of the RUL predictions over engine life
    var_pred_lst = BNN_results['var'] #Variance of the RUL predictions over engine life
    true_lst = BNN_results['true'] #Ground truth RUL over engine life

    upper_90 = np.array([plot_mean_and_percentile(mean_pred_lst[i], var_pred_lst[i], percentile=90, upper_lower='upper') for i in range(len(mean_pred_lst))])
    lower_90 = np.array([plot_mean_and_percentile(mean_pred_lst[i], var_pred_lst[i], percentile=90, upper_lower='lower') for i in range(len(mean_pred_lst))])

    with open(DNN_engines[engine_id], 'r') as jsonfile:
        DNN_results = json.load(jsonfile)
    
    #Deterministic results
    DNN_pred_lst = DNN_results['pred'] #RUl predictions over engine life

    BNN_error = [(mean_pred_lst[i] - true_lst[i]) for i in range(len(true_lst))] #BNN error
    DNN_error = [(DNN_pred_lst[i] - true_lst[i]) for i in range(len(true_lst))]#DNN error
    B_RMSE = np.round(np.sqrt(np.mean([(mean_pred_lst[i] - true_lst[i])**2 for i in range(len(true_lst))])), 2) #Root Mean Squared error of Bayesian prediciton
    D_RMSE = np.round(np.sqrt(np.mean([(DNN_pred_lst[i] - true_lst[i])**2 for i in range(len(true_lst))] )), 2) #Root Mean Squared error of Deterministic prediciton

    cycles = len(mean_pred_lst)

    BNN_df = pd.DataFrame({
    'Mean': mean_pred_lst,
    'RUL': true_lst,
    'Lower Bound': lower_90,
    'Upper Bound': upper_90,
    'Cycle': range(cycles)
})
    
    DNN_df = pd.DataFrame({
    'Mean': DNN_pred_lst,
    'RUL': true_lst,
    'Cycle': range(cycles)
})


    # Alternatively, for a clearer distinction between mean, bounds, and true values:
    plt.figure(figsize=(6, 2.2))
    sns.lineplot(x='Cycle', y='RUL', data=BNN_df, color='red', label='RUL')
    sns.lineplot(x='Cycle', y='Mean', data=BNN_df, color='blue', label=f'Bayesian Mean Prediction.')
    sns.lineplot(x='Cycle', y='Mean', data=DNN_df, color='orange', label=f'Deterministic Prediction.', linestyle='--')
    plt.fill_between(BNN_df['Cycle'], BNN_df['Lower Bound'], BNN_df['Upper Bound'], color='blue', alpha=0.2, label='90% Prediction Interval')
    plt.legend()
    # plt.title(f'BNN RUL prediction of test engine {engine_id + 1}')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    # Save the figure
    # plt.savefig(f'/Users/jillesandringa/Documents/AE/MSc/Thesis/Graphs_and_figures/prediction_interval_plot_{noisy}.pdf', format='pdf', bbox_inches='tight')
    print(f'Bayesian RMSE: {B_RMSE}, Deterministic RMSE: {D_RMSE}')
    plt.show()