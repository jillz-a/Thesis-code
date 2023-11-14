#%%Vizualizing script for the BNN model including DNN and uncertainty predictions
import os
import glob
import csv
import numpy as np

import plotly.offline as pyo
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import norm

from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt

from torch import load
from tqdm import tqdm
import torch
from torchvision.transforms import ToTensor
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

engine_eval = 0
#%%
#import BNN results: every file represents 1 engine
BNN_result_path = os.path.join(project_path, 'BNN/BNN_results', DATASET)
engines= glob.glob(os.path.join(BNN_result_path, '*.json'))  # Get a list of all file paths in the folder
engines.sort() 

#import CF results: every file represents 1 engine
CF_result_path = os.path.join(project_path, 'DiCE/BNN_cf_results/outputs', DATASET)
CF_engines= glob.glob(os.path.join(CF_result_path, '*.json'))  # Get a list of all file paths in the folder
CF_engines.sort() 

#import DNN results: every file represents 1 engine
DNN_result_path = os.path.join(project_path, 'BNN/DNN_results', DATASET)
DNN_engines= glob.glob(os.path.join(DNN_result_path, '*.json'))  # Get a list of all file paths in the folder
DNN_engines.sort() 

for engine in engines[engine_eval: engine_eval+1]:
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

        
    #%% Plot data
    BNN_error = [(mean_pred_lst[i] - true_lst[i]) for i in range(len(true_lst))] #BNN error
    DNN_error = [(y_pred_lst[i] - true_lst[i]) for i in range(len(true_lst))]#DNN error
    B_RMSE = np.round(np.sqrt(np.mean([(mean_pred_lst[i] - true_lst[i])**2 for i in range(len(true_lst))])), 2) #Root Mean Squared error of Bayesian prediciton
    D_RMSE = np.round(np.sqrt(np.mean([(y_pred_lst[i] - true_lst[i])**2 for i in range(len(true_lst))] )), 2) #Root Mean Squared error of Deterministic prediciton