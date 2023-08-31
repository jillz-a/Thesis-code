#%%Vizualizing script for the ML models
import os
import glob
import csv
import numpy as np
import matplotlib.pyplot as plt

import plotly.express as px
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from torch import nn, load
from tqdm import tqdm
import torch
from torchvision.transforms import ToTensor
import time
import sys

# Get the absolute path of the project directory
project_path = os.path.dirname(os.path.abspath(os.path.join((__file__), os.pardir)))
# Add the project directory to sys.path if it's not already present
if project_path not in sys.path:
    sys.path.append(project_path)

from DNN import NeuralNetwork
from BNN import BayesianNeuralNetwork
from DNN_vizualizer import y_pred_lst, D_RMSE

from variables import *

start = time.time()

folder_path = f'data/{DATASET}/min-max/test'  # Specify the path to your folder

with open(os.path.join(project_path, folder_path, '0-Number_of_samples.csv')) as csvfile:
    sample_len = list(csv.reader(csvfile)) #list containing the amount of samples per engine/trajectory

file_paths = glob.glob(os.path.join(folder_path, '*.txt'))  # Get a list of all file paths in the folder
file_paths.sort() 

#%%
engines = [8]
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

    #Go through each sample
    loop = tqdm(selected_file_paths)
    for file_path in loop:
    
        # Process each selected file
        sample = np.genfromtxt(file_path, delimiter=" ", dtype=np.float32)
        label = float(file_path[-7:-4])

        #Import into trained machine learning models
        NNmodel = BayesianNeuralNetwork(input_size, hidden_size).to(device)
        with open(f'{project_path}/BNN/BNN_model_state_{DATASET}_test.pt', 'rb') as f: 
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

    
#%% Plot data
error = [(mean_pred_lst[i] - true_lst[i])**2 for i in range(len(true_lst))]
B_RMSE = np.round(np.sqrt(np.mean(error)), 2)

fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1)
#mean RUL predictions from Bayesian model
fig.add_trace(go.Scatter(x=np.arange(len(mean_pred_lst)), 
                            y=mean_pred_lst, 
                            name=f'Bayesian Mean Predicted RUL values for engine {engine}, RMSE = {B_RMSE}',
                            mode= 'lines'), row=1, col=1)

#deterministic RUL predictions
fig.add_trace(go.Scatter(x=np.arange(len(y_pred_lst)),
                            y=y_pred_lst,
                            name= f'Deterministic Predicted RUL values, RMSE = {D_RMSE}',
                            mode='lines'), row=1, col=1)

#true RUL values
fig.add_trace(go.Scatter(x=np.arange(len(true_lst)),
                            y=true_lst,
                            name='True RUL values',
                            mode='lines'), row=1, col=1)


#1 standard deviation interval
fig.add_trace(go.Scatter(
    x=np.concatenate((np.arange(len(mean_pred_lst)), np.arange(len(mean_pred_lst))[::-1])),  # Concatenate x values in both directions
    y=np.concatenate((np.array(mean_pred_lst) + np.sqrt(var_pred_lst), np.array(mean_pred_lst)[::-1] - np.sqrt(var_pred_lst)[::-1])),
    fill='toself',  # Fill to next y values
    fillcolor='rgba(0, 100, 80, 0.2)',  # Color of the filled area
    line=dict(color='rgba(255, 255, 255, 0)'),  # Hide the line
    name='1 Standard Deviation'
), row=1, col=1)

#2 standard deviation interval
fig.add_trace(go.Scatter(
x=np.concatenate((np.arange(len(mean_pred_lst)), np.arange(len(mean_pred_lst))[::-1])),  # Concatenate x values in both directions
y=np.concatenate((np.array(mean_pred_lst) + 2*np.sqrt(var_pred_lst), np.array(mean_pred_lst)[::-1] - 2*np.sqrt(var_pred_lst)[::-1])),
fill='toself',  # Fill to next y values
fillcolor='rgba(0, 90, 80, 0.2)',  # Color of the filled area
line=dict(color='rgba(255, 255, 255, 0)'),  # Hide the line
name='2 Standard Deviation'
), row=1, col=1)


#alpha-lambda accuracy
alpha = 0.2
Lambda = 0.75
fig.add_trace(go.Scatter(
x=np.concatenate((np.arange(len(mean_pred_lst)), np.arange(len(mean_pred_lst))[::-1])),  # Concatenate x values in both directions
y=np.concatenate((np.array([i*(1.0+alpha) for i in true_lst]), np.array([i*(1.0-alpha) for i in true_lst])[::-1])),
fill='toself',  # Fill to next y values
fillcolor='rgba(0, 80, 200, 0.15)',  # Color of the filled area
line=dict(color='rgba(255, 255, 255, 0)'),  # Hide the line
name=f'\u03B1 +-{alpha*100}%, \u03BB {Lambda}'
), row=1, col=1)


fig.add_shape(type='line', x0=np.round(len(mean_pred_lst)*Lambda), x1=np.round(len(mean_pred_lst)*Lambda), y0=0, y1=true_lst[int(np.round(len(mean_pred_lst)*Lambda))], line_dash='dash', row=1, col=1)

#sub plot of the propability distribution of the RUL prediction
def create_sub_plot(mean, std_dev):
    x_sub = np.linspace(mean - 3 * std_dev, mean + 3 * std_dev, 100)
    y_sub = (1 / (std_dev * np.sqrt(2 * np.pi))) * np.exp(-(x_sub - mean)**2 / (2 * std_dev**2))
    
    sub_trace = go.Scatter(
        x=x_sub,
        y=y_sub,
        mode='lines',
        name='Normal Distribution',
        line=dict(color='blue')
    )
    
    return sub_trace

fig.add_trace(go.Scatter(), row=2, col=1)

# Update subplot on hover
def update_subplot(trace, points, selector):
    if len(points.point_inds) > 0:
        ind = points.point_inds[0]
        sub_trace = create_sub_plot(mean_pred_lst[ind], np.sqrt(var_pred[ind]))
        fig.update_traces(sub_trace, selector=dict(name='Normal Distribution'))

fig.data[0].on_hover(update_subplot)


finish = time.time()
print(f'elapsed time = {finish - start} seconds')

fig.update_layout(
    title=f'Dataset {DATASET}, {n_samples} samples per data point, average variance = {np.round(np.mean(var_pred_lst),2)}',
    xaxis_title="Cycles",
    yaxis_title="RUL",
    hovermode='closest'
)
fig.show()
