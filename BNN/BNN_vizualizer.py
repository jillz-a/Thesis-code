#%%Vizualizing script for the ML models
import os
import glob
import csv
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
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

fig = go.Figure()
#mean RUL predictions from Bayesian model
fig.add_trace(go.Scatter(x=np.arange(len(mean_pred_lst)), 
                            y=mean_pred_lst, 
                            name=f'Bayesian Mean Predicted RUL values for engine {engine}, RMSE = {B_RMSE}',
                            mode= 'lines'))
# plt.plot(mean_pred_lst, label= f'Bayesian Mean Predicted RUL values for engine {engine}, RMSE = {B_RMSE}')
#deterministic RUL predictions
fig.add_trace(go.Scatter(x=np.arange(len(y_pred_lst)),
                            y=y_pred_lst,
                            name= f'Deterministic Predicted RUL values, RMSE = {D_RMSE}',
                            mode='lines'))
# plt.plot(y_pred_lst, label=f'Deterministic Predicted RUL values, RMSE = {D_RMSE}')
#true RUL values
fig.add_trace(go.Scatter(x=np.arange(len(true_lst)),
                            y=true_lst,
                            name='True RUL values',
                            mode='lines'))
# plt.plot(true_lst, label='True RUL values')

#1 standard deviation interval
fig.add_trace(go.Scatter(
    x=np.concatenate((np.arange(len(mean_pred_lst)), np.arange(len(mean_pred_lst))[::-1])),  # Concatenate x values in both directions
    y=np.concatenate((np.array(mean_pred_lst) + np.sqrt(var_pred_lst), np.array(mean_pred_lst)[::-1] - np.sqrt(var_pred_lst)[::-1])),
    fill='toself',  # Fill to next y values
    fillcolor='rgba(0, 100, 80, 0.2)',  # Color of the filled area
    line=dict(color='rgba(255, 255, 255, 0)'),  # Hide the line
    name='1 Standard Deviation'
))

# plt.fill_between(x=np.arange(len(mean_pred_lst)), 
#                  y1= mean_pred_lst + np.sqrt(var_pred_lst), 
#                  y2=mean_pred_lst - np.sqrt(var_pred_lst),
#                  alpha= 0.5,
#                 #  color = 'yellow',
#                  label= '1 STD interval'
#                  )
#2 standard deviation interval
fig.add_trace(go.Scatter(
x=np.concatenate((np.arange(len(mean_pred_lst)), np.arange(len(mean_pred_lst))[::-1])),  # Concatenate x values in both directions
y=np.concatenate((np.array(mean_pred_lst) + 2*np.sqrt(var_pred_lst), np.array(mean_pred_lst)[::-1] - 2*np.sqrt(var_pred_lst)[::-1])),
fill='toself',  # Fill to next y values
fillcolor='rgba(0, 90, 80, 0.2)',  # Color of the filled area
line=dict(color='rgba(255, 255, 255, 0)'),  # Hide the line
name='2 Standard Deviation'
))

# plt.fill_between(x=np.arange(len(mean_pred_lst)), 
#                  y1= mean_pred_lst + 2*np.sqrt(var_pred_lst), 
#                  y2=mean_pred_lst - 2*np.sqrt(var_pred_lst),
#                  alpha= 0.3,
#                 #  color = 'yellow',
#                  label= '2 STD interval'
#                  )

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
))

# plt.fill_between(x=np.arange(len(mean_pred_lst)), 
#                  y1=[i*(1.0+alpha) for i in true_lst], 
#                  y2=[i*(1.0-alpha) for i in true_lst], 
#                  alpha= 0.3,
#                  label=f'\u03B1 +-{alpha*100}%, \u03BB {Lambda}'
#                  )

fig.add_shape(type='line', x0=np.round(len(mean_pred_lst)*Lambda), x1=np.round(len(mean_pred_lst)*Lambda), y0=0, y1=true_lst[int(np.round(len(mean_pred_lst)*Lambda))], line_dash='dash')
# plt.vlines(x= np.round(len(mean_pred_lst)*Lambda), ymin=0, ymax=true_lst[int(np.round(len(mean_pred_lst)*Lambda))], linestyles='dashed')
#%%

finish = time.time()
print(f'elapsed time = {finish - start} seconds')

fig.update_layout(
    title=f'Dataset {DATASET}, {n_samples} samples per data point, average variance = {np.round(np.mean(var_pred_lst),2)}',
    xaxis_title="Cycles",
    yaxis_title="RUL",
)
fig.show()

# plt.xlabel('Cycles')
# plt.ylabel('RUL')
# plt.grid()
# plt.title(f'Dataset {DATASET}, {n_samples} samples per data point, average variance = {np.round(np.mean(var_pred_lst),2)}')
# plt.legend()
# plt.show()