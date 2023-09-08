#%%Vizualizing script for the ML models
import os
import glob
import csv
import numpy as np

import plotly.offline as pyo
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import norm

from torch import load
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

#definitions
def alpha_dist(lower_bound, upper_bound, mean, stdev):
            prob_lower = norm.cdf(lower_bound, loc=mean, scale=stdev)
            prob_upper = norm.cdf(upper_bound, loc=mean, scale=stdev)

            percentage_in_range = (prob_upper - prob_lower)

            return percentage_in_range

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
x_plot = np.arange(len(mean_pred_lst))
alpha = 0.2

# Create a subplot with 2 rows and 1 column for the sub-plot and table
fig = make_subplots(rows=2, cols=2, shared_xaxes=False, vertical_spacing=0.1,
                    subplot_titles=[f'RUL prediction of engine {engine}', f'Distribution within \u03B1 +-{alpha*100}%', 'Prediction distribution'])

plot_figure = [
                go.Scatter(x=x_plot, 
                           y=mean_pred_lst, 
                           mode='lines', 
                           line=dict(color='blue'),
                           visible=False,
                           name=f'Bayesian Mean Predicted RUL values for engine {engine}, RMSE = {B_RMSE}'),
                go.Scatter(x=x_plot,
                            y=y_pred_lst,
                            name= f'Deterministic Predicted RUL values, RMSE = {D_RMSE}',
                            mode='lines',
                            visible=False,
                            line=dict(color='orange')),
                go.Scatter(x=x_plot,
                            y=true_lst,
                            name='True RUL values',
                            mode='lines',
                            visible=False,
                            line=dict(color='red')),
                go.Scatter(x=np.concatenate((x_plot, x_plot[::-1])), 
                            y=np.concatenate((np.array(mean_pred_lst) + np.sqrt(var_pred_lst), np.array(mean_pred_lst)[::-1] - np.sqrt(var_pred_lst)[::-1])),
                            fill='toself',  # Fill to next y values
                            fillcolor='rgba(0, 100, 80, 0.2)',  # Color of the filled area
                            line=dict(color='rgba(255, 255, 255, 0)'),  # Hide the line
                            visible=False,
                            name='1 Standard Deviation',
                            hoverinfo='skip'),
                go.Scatter(x=np.concatenate((x_plot, x_plot[::-1])), 
                            y=np.concatenate((np.array(mean_pred_lst) + 2*np.sqrt(var_pred_lst), np.array(mean_pred_lst)[::-1] - 2*np.sqrt(var_pred_lst)[::-1])),
                            fill='toself',  # Fill to next y values
                            fillcolor='rgba(0, 90, 80, 0.2)',  # Color of the filled area
                            line=dict(color='rgba(255, 255, 255, 0)'),  # Hide the line
                            visible=False,
                            name='2 Standard Deviation',
                            hoverinfo='skip'),
                go.Scatter(x=np.concatenate((x_plot, x_plot[::-1])), 
                            y=np.concatenate((np.array([i*(1.0+alpha) for i in true_lst]), np.array([i*(1.0-alpha) for i in true_lst])[::-1])),
                            fill='toself',  # Fill to next y values
                            fillcolor='rgba(0, 80, 200, 0.15)',  # Color of the filled area
                            visible=False,
                            line=dict(color='rgba(255, 255, 255, 0)'),  # Hide the line
                            name=f'\u03B1 +-{alpha*100}%, \u03BB',
                            hoverinfo='skip')
            ]
            

std_dev = np.sqrt(var_pred_lst)
for i in range(len(x_plot)):
    for j in range(len(plot_figure)):
        fig.add_trace(plot_figure[j], row=1, col=1)

    
    y_sub = np.linspace(mean_pred_lst[i] - 3 * std_dev[i], mean_pred_lst[i] + 3 * std_dev[i], 100)
    x_sub = norm.pdf(y_sub, mean_pred_lst[i], std_dev[i])

    fig.add_trace(go.Scatter(x=x_sub, 
                            y=y_sub, 
                            visible=False,
                            mode='lines', 
                            line=dict(color='rgba(0, 100, 80, 0.2)'),
                            fill='tozeroy',
                            name='RUL prediction distribution'),
                            row=2,
                            col=1)
    
    fig.add_trace(go.Scatter(x=np.array([0, max(x_sub)]),
                            y=np.array([true_lst[i], true_lst[i]]),
                            visible=False,
                            mode='lines',
                            line=dict(dash='dash', color='blue'),
                            name='True RUL'),
                            row=2,
                            col=1)
    
    fig.add_trace(go.Scatter(x=np.array([0, max(x_sub)]),
                            y=np.array([y_pred_lst[i], y_pred_lst[i]]),
                            visible=False,
                            mode='lines',
                            line=dict(dash='dash', color='orange'),
                            name='Deterministic Predicted RUL'),
                            row=2,
                            col=1)
    
    fig.add_trace(go.Scatter(x=np.array([i, i]),
                            y=np.array([0, mean_pred_lst[i]]),
                            visible=False,
                            mode='lines',
                            line=dict(dash='dash', color='black'),
                            showlegend=False),
                            row=1,
                            col=1)
    
    fig.add_trace(go.Scatter(x=np.array([0, 0]),
                            y=np.array([0, 140]),
                            visible=False,
                            mode='lines',
                            line=dict(color='rgba(255, 255, 255, 0)'),
                            showlegend=False),
                            row=2,
                            col=1)
    fig.add_trace(go.Scatter(x=np.array([0, 0.55]),
                            y=np.array([0, 0]),
                            visible=False,
                            mode='lines',
                            line=dict(color='rgba(255, 255, 255, 0)'),
                            showlegend=False),
                            row=2,
                            col=1)
    
    fig.add_trace(go.Scatter(x=x_plot,
                                y = np.array([alpha_dist(true_lst[i]*(1-alpha), true_lst[i]*(1+alpha), mean_pred_lst[i], np.sqrt(var_pred_lst[i])) for i in range(len(x_plot))]),
                                visible=False,
                                mode='lines',
                                fill='tozerox',
                                line=dict(color='rgba(0, 80, 200, 0.5)'),
                                name=f'Distribution within \u03B1 +-{alpha*100}%'),
                                row=1,
                                col=2)


n_traces = int(len(fig.data)/(max(x_plot)+1))
for i in range(n_traces):
    fig.data[i].visible = True

steps = []
for i in range(len(x_plot)):
    step = dict(
        method="update",
        args=[{"visible": [False] * len(fig.data)},
              {"title": "Slider switched to step: " + str(i)}],  # layout attribute
    )
    for j in range(i*n_traces, (i+1)*n_traces):
        step["args"][0]["visible"][j] = True  # Toggle i'th trace to "visible"
    steps.append(step)

sliders = [dict(
    active=0,
    currentvalue={"prefix": "Cycle: "},
    pad={"t": 50},
    steps=steps
)]

# xaxis_main = go.layout.XAxis()
# xaxis_sub = go.layout.XAxis()
# yaxis_sub = go.layout.YAxis()

# xaxis_sub.update(range=[0, 0.6], ticklabelposition='outside bottom')
# yaxis_sub.update(range=[0, 140])


fig.update_layout(
    sliders=sliders,
    title=f'RUL prediction of engine {engine}', 
    showlegend=True,
    # xaxis1=xaxis_main,
    # xaxis2=xaxis_sub,
    # yaxis2 = yaxis_sub
)

fig.show()

# Export the figure to an HTML file
pyo.plot(fig, filename='interactive-plots/prediction_fig.html', auto_open=False)


# %%
