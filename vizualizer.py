#%%Vizualizing script for the ML models
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from torch import nn, load
from torchvision.transforms import ToTensor

from BNN.BNN import NeuralNetwork

device = 'mps'

#import data
folder_path = 'data/FD001/min-max/train'  # Specify the path to your folder
num_files_to_select = 162  # Specify the number of files to select

file_paths = glob.glob(os.path.join(folder_path, '*.txt'))  # Get a list of all file paths in the folder
file_paths.sort() 

selected_file_paths = file_paths[:num_files_to_select]  # Select the desired number of files

#setup data to plot
y_pred_lst = []
y_lst = []

#%%Go through each sample
for file_path in selected_file_paths:
    # Process each selected file
    sample = np.genfromtxt(file_path, delimiter=" ", dtype=np.float32)
    label = float(file_path[-7:-4])
    print(file_path, label)

    #Import into trained machine learning models
    NNmodel = NeuralNetwork().to(device)
    with open('BNN/model_state.pt', 'rb') as f: 
            NNmodel.load_state_dict(load(f)) 

    #predict RUL from samples
    X = ToTensor()(sample).unsqueeze(0).to(device)
    y_pred = NNmodel(X)
    y_pred = y_pred.to('cpu')
    y_pred = y_pred.detach().numpy()

    y = label #True RUL

    #add predictions and true labels to lists
    y_pred_lst.append(y_pred.item())
    y_lst.append(y)
#%%

plt.plot(y_pred_lst)
plt.plot(y_lst)
plt.show()