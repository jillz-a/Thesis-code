import os
import numpy as np
import torch
from torch import nn, save, load
from torch.utils.data import DataLoader
from torch.optim import Adam

device = 'mps'
#Frequentist neural network class
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(30*14, 5),
            nn.ReLU(),
            nn.Linear(5, 1),
        )

    def forward(self, x):
        # x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
    
#Model, optimizer and loss function
NNmodel = NeuralNetwork().to(device)
opt = Adam(NNmodel.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()

#training and validation data
train = DataLoader()