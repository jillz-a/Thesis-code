#%%Vizualizing script for the ML models
import os
import glob
import csv
import numpy as np
import matplotlib.pyplot as plt

import plotly.io as pio
import dash
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import norm

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
x_plot = np.arange(len(mean_pred_lst))
alpha = 0.2

app = dash.Dash(__name__)

plot_figure = {
            'data': [
                go.Scatter(x=x_plot, 
                           y=mean_pred_lst, 
                           mode='lines', 
                           name=f'Bayesian Mean Predicted RUL values for engine {engine}, RMSE = {B_RMSE}'),
                go.Scatter(x=x_plot,
                            y=y_pred_lst,
                            name= f'Deterministic Predicted RUL values, RMSE = {D_RMSE}',
                            mode='lines'),
                go.Scatter(x=x_plot,
                            y=true_lst,
                            name='True RUL values',
                            mode='lines'),
                go.Scatter(x=np.concatenate((x_plot, x_plot[::-1])), 
                            y=np.concatenate((np.array(mean_pred_lst) + np.sqrt(var_pred_lst), np.array(mean_pred_lst)[::-1] - np.sqrt(var_pred_lst)[::-1])),
                            fill='toself',  # Fill to next y values
                            fillcolor='rgba(0, 100, 80, 0.2)',  # Color of the filled area
                            line=dict(color='rgba(255, 255, 255, 0)'),  # Hide the line
                            name='1 Standard Deviation',
                            hoverinfo='skip'),
                go.Scatter(x=np.concatenate((x_plot, x_plot[::-1])), 
                            y=np.concatenate((np.array(mean_pred_lst) + 2*np.sqrt(var_pred_lst), np.array(mean_pred_lst)[::-1] - 2*np.sqrt(var_pred_lst)[::-1])),
                            fill='toself',  # Fill to next y values
                            fillcolor='rgba(0, 90, 80, 0.2)',  # Color of the filled area
                            line=dict(color='rgba(255, 255, 255, 0)'),  # Hide the line
                            name='2 Standard Deviation',
                            hoverinfo='skip'),
                go.Scatter(x=np.concatenate((x_plot, x_plot[::-1])), 
                            y=np.concatenate((np.array([i*(1.0+alpha) for i in true_lst]), np.array([i*(1.0-alpha) for i in true_lst])[::-1])),
                            fill='toself',  # Fill to next y values
                            fillcolor='rgba(0, 80, 200, 0.15)',  # Color of the filled area
                            line=dict(color='rgba(255, 255, 255, 0)'),  # Hide the line
                            name=f'\u03B1 +-{alpha*100}%, \u03BB',
                            hoverinfo='skip'),
            ],
            'layout': {'title': f'RUL prediction of engine {engine}'}
        }

app.layout = html.Div([
    dcc.Graph(
        id='main-plot',
        figure=plot_figure
    ),
    html.Div(
        style={'display': 'flex', 'flexDirection': 'row'},
        children=[
            dcc.Graph(id='sub-plot'),
            html.Div(id='table-container')
        ]
    ),
    html.Button('Export Plot', id='export-button'),
    html.Div(id='export-message')
])

@app.callback(
    Output('sub-plot', 'figure'),
    Output('table-container', 'children'),
    Input('main-plot', 'hoverData')
)
def display_sub_plot_and_table(hover_data):
    if hover_data:
        x_selected = hover_data['points'][0]['x']
        hover_mean = mean_pred_lst[x_selected]
        std_dev = np.sqrt(var_pred_lst[x_selected])
        
        y_sub = np.linspace(hover_mean - 3 * std_dev, hover_mean + 3 * std_dev, 100)
        x_sub = norm.pdf(y_sub, hover_mean, std_dev)
        
        sub_trace = go.Scatter(x=x_sub, 
                               y=y_sub, 
                               mode='lines', 
                               fill= 'tozeroy',
                               name='RUL prediction distribution')
        true_trace = go.Scatter(x=np.array([0, max(x_sub)]),
                                y=np.array([true_lst[x_selected], true_lst[x_selected]]),
                                mode='lines',
                                line_dash='dash',
                                name='True RUL')
        
        sub_layout = go.Layout(title='Prediction Distribution at cycle {}'.format(x_selected))
        
        # Rotate the sub-plot by swapping x and y axes and updating labels
        sub_layout.xaxis = go.layout.XAxis(title='Density')
        sub_layout.yaxis = go.layout.YAxis(title='Cycles')
        
        sub_fig = {'data': [sub_trace, true_trace], 'layout': sub_layout}
        def alphalambda():
            if hover_mean < true_lst[x_selected]*(1+alpha) and hover_mean > true_lst[x_selected]*(1-alpha):
                return 1
            else:
                return 0
            
        def alpha_dist(lower_bound, upper_bound):
            prob_lower = norm.cdf(lower_bound, loc=hover_mean, scale=std_dev)
            prob_upper = norm.cdf(upper_bound, loc=hover_mean, scale=std_dev)

            percentage_in_range = (prob_upper - prob_lower)*100

            return percentage_in_range
        
        # Create a table with some example values
        table_data = [{'Metric': 'Mean', 'Value': np.round(hover_mean,2), 'Unit': 'Cycles'},
                      {'Metric': 'Standard Deviation', 'Value': np.round(std_dev,2), 'Unit': 'Cycles'},
                      {'Metric': 'Bayesian RMSE', 'Value': B_RMSE, 'Unit': 'Cycles'},
                      {'Metric': 'Deterministic RMSE', 'Value': D_RMSE, 'Unit': 'Cycles'},
                      {'Metric': f'\u03B1 +-{alpha*100}%, \u03BB {np.round(x_selected/max(x_plot),2)}', 'Value': alphalambda(), 'Unit': '-'},
                      {'Metric': 'Distribution within \u03B1 bounds', 'Value': np.round(alpha_dist(true_lst[x_selected]*(1-alpha), true_lst[x_selected]*(1+alpha)),1), 'Unit': '%'}]
        table = dash_table.DataTable(
            columns=[{'name': col, 'id': col} for col in table_data[0].keys()],
            data=table_data
        )
        
        return sub_fig, table
    
    return {'data': [], 'layout': {}}, None

#export button to html
@app.callback(
    Output('export-message', 'children'),
    Input('export-button', 'n_clicks')
)
def export_plot(n_clicks):
    if n_clicks is not None:
        pio.write_html(app.layout, 'interactive-plots/dash_layout.html')
        return html.P('Layout exported to dash_layout.html')
    return html.P('')


if __name__ == "__main__":
    app.run_server(debug=False)
