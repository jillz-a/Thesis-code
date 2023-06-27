#%% import dependencies
import os
import glob
import sys
import numpy as np
import torch
from tqdm import tqdm

from torch import nn, save, load
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import LambdaLR
from torch.optim import Adam
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt

from bayesian_torch.models.dnn_to_bnn import dnn_to_bnn, get_kl_loss
from BNN.DNN import NeuralNetwork

device = 'cpu'

#Import into trained machine learning models
input_size = 14
hidden_size = 32
num_layers = 1

NNmodel = NeuralNetwork(input_size, hidden_size, num_layers).to(device)

with open('BNN/model_state.pt', 'rb') as f: 
    NNmodel.load_state_dict(load(f)) 