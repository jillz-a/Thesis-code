#%%Vizualizing script for the ML models
import os
import glob
import csv
import numpy as np
import matplotlib.pyplot as plt
from torch import nn, load
from torchvision.transforms import ToTensor

from BNN.DNN import NeuralNetwork

device = 'cpu'

#import data
DATASET = 'FD002'
folder_path = f'data/{DATASET}/min-max/train'  # Specify the path to your folder

with open(os.path.join(folder_path, '0-Number_of_samples.csv')) as csvfile:
    sample_len = list(csv.reader(csvfile)) #list containing the amount of samples per engine/trajectory

file_paths = glob.glob(os.path.join(folder_path, '*.txt'))  # Get a list of all file paths in the folder
file_paths.sort() 

index = 0
for engine in range(5):
    selected_file_paths = file_paths[index:index + int(sample_len[engine][0])]  # Select the desired number of files
    index += int(sample_len[engine][0])

    #setup data to plot
    y_pred_lst = []
    y_lst = []

    # Model input parameters
    input_size = 14 #number of features
    hidden_size = 32
    num_layers = 1

    #%%Go through each sample
    for file_path in selected_file_paths:
        # Process each selected file
        sample = np.genfromtxt(file_path, delimiter=" ", dtype=np.float32)
        label = float(file_path[-7:-4])

        #Import into trained machine learning models
        NNmodel = NeuralNetwork(input_size, hidden_size, num_layers).to(device)
        with open(f'BNN/model_state_{DATASET}.pt', 'rb') as f: 
            NNmodel.load_state_dict(load(f)) 

        #predict RUL from samples
        X = ToTensor()(sample).to(device)
        y_pred = NNmodel(X)
        y_pred = y_pred.to('cpu')
        y_pred = y_pred.detach().numpy()

        y = label #True RUL

        #add predictions and true labels to lists
        y_pred_lst.append(y_pred.item())
        y_lst.append(y)

    error = [y_pred_lst[i] - y_lst[i] for i in range(len(y_lst))]

    plt.plot(y_pred_lst, label= 'Predicted RUL values')
    plt.plot(y_lst, label='True RUL values')
#%%


plt.xlabel('Cycles')
plt.ylabel('RUL')
# plt.legend()
plt.show()