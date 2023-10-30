#%% import dependencies
import glob
import sys
import os
import csv
import json

# Get the absolute path of the project directory
project_path = os.path.dirname(os.path.abspath(os.path.join((__file__), os.pardir)))
# Add the project directory to sys.path if it's not already present
if project_path not in sys.path:
    sys.path.append(project_path)

import numpy as np
import torch
from tqdm import tqdm
from sklearn.model_selection import KFold
import pandas as pd

from torch import nn, save, load
from torch.utils.data import DataLoader, ConcatDataset, SequentialSampler, random_split
from torch.optim.lr_scheduler import LambdaLR
from torch.optim import Adam
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt

from bayesian_torch.models.dnn_to_bnn import get_kl_loss
import bayesian_torch.layers as bl

from CLUE_master.VAE.fc_gauss import VAE_gauss_net
from CLUE_master.VAE.train import train_VAE


from variables import *

torch.manual_seed(42)
current_directory = os.getcwd()  # Get the current working directory
parent_directory = os.path.abspath(os.path.join(current_directory, os.pardir))  # Get the absolute path of the parent directory

TRAINDATASET = os.path.abspath(os.path.join(parent_directory, f'Thesis Code/data/{DATASET}/min-max/train'))
TESTDATASET = os.path.abspath(os.path.join(parent_directory, f'Thesis Code/data/{DATASET}/min-max/test'))

save_dir = os.path.join(project_path, f'/CLUE/VAE_model_states/{DATASET}/VAE_model_state_test')

#%% main script
if __name__ == '__main__':

    from BNN.Data_loader import CustomDataset
    train = CustomDataset(TRAINDATASET)
    test = CustomDataset(TESTDATASET)

    train_set, val_set = random_split(train, [0.8, 0.2])

    train_data = DataLoader(train_set)
    val_data = DataLoader(val_set, batch_size= len(val_set))

    batch_size = 128
    nb_epochs = 2500
    early_stop = 200
    lr = 1e-4
    cuda = torch.cuda.is_available()

    net = VAE_gauss_net(input_dim=train_data.size(), width=300, depth=3, latent_dim=2, pred_sig=False, lr=lr)

    vlb_train, vlb_dev = train_VAE(net, save_dir, batch_size, nb_epochs, train_data, val_data,
                                       cuda=cuda, flat_ims=False, train_plot=False, early_stop=early_stop)