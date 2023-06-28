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
from DNN import NeuralNetwork

torch.manual_seed(42)
current_directory = os.getcwd()  # Get the current working directory
parent_directory = os.path.abspath(os.path.join(current_directory, os.pardir))  # Get the absolute path of the parent directory

device = 'cpu'
DATASET = 'FD001'
TRAINDATASET = os.path.abspath(os.path.join(parent_directory, f'Thesis Code/data/{DATASET}/min-max/train'))
TESTDATASET = os.path.abspath(os.path.join(parent_directory, f'Thesis Code/data/{DATASET}/min-max/test'))
BATCHSIZE = 100
EPOCHS = 10
TRAIN = True

#%% Import into trained machine learning models
input_size = 14
hidden_size = 32
num_layers = 1

NNmodel = NeuralNetwork(input_size, hidden_size, num_layers).to(device)

# with open(f'BNN/model_state_{DATASET}.pt', 'rb') as f: 
#     NNmodel.load_state_dict(load(f)) 

#%% main script
if __name__ == '__main__':

    const_bnn_prior_parameters = {
        "prior_mu": 0.0,
        "prior_sigma": 1.0,
        "posterior_mu_init": 0.0,
        "posterior_rho_init": -3.0,
        "type": "Reparameterization",  # Flipout or Reparameterization
        "moped_enable": False,  # True to initialize mu/sigma from the pretrained dnn weights
        "moped_delta": 0.5,
}

    dnn_to_bnn(NNmodel, const_bnn_prior_parameters)

    from Data_loader import CustomDataset
    train = CustomDataset(TRAINDATASET)
    test = CustomDataset(TESTDATASET)
    train_data = DataLoader(train, batch_size=BATCHSIZE)
    test_data = DataLoader(test, batch_size=BATCHSIZE)

    opt = Adam(NNmodel.parameters(), lr=1e-3)

    loss_fn = nn.MSELoss()

    # Define the lambda function for decaying the learning rate
    lr_lambda = lambda epoch: (1 - min(60-1, epoch) / 60) ** 0.7
    # Create the learning rate scheduler
    scheduler = LambdaLR(opt, lr_lambda=lr_lambda)

    #%% Train the model
    loss_lst = []
    if TRAIN == True:
        #%%
        print(f"Training model: {DATASET}")
        print(f"Batch size: {BATCHSIZE}, Epochs: {EPOCHS}")
        for epoch in range(EPOCHS):
            loop = tqdm(train_data)
            for batch in loop:
                X, y = batch #Input sample, true RUL
                y = torch.t(y) #Transpose to fit X dimension
                X, y = X.to(device), y.to(device) #send to device
                print(np.shape(X))
                y_pred = NNmodel(X) #Run model
                kl = get_kl_loss(NNmodel)
                print(kl)
                ce_loss = torch.sqrt(loss_fn(y_pred[:,0], y)) #RMSE loss function
                loss = ce_loss + kl / BATCHSIZE
                # print(y_pred, y_pred[:,0],y, loss)
                

            
                #Backprop
                opt.zero_grad()
                loss.backward()
                opt.step()
            
                loop.set_description(f"Epoch: {epoch+1}/{EPOCHS}")
                loop.set_postfix(loss = np.sqrt(loss.item()), lr = opt.param_groups[0]['lr'])

            scheduler.step() 
            loss_lst.append(loss.item())  

        with open(f'BNN/BNN_model_state_{DATASET}.pt', 'wb') as f:
            save(NNmodel.state_dict(), f)
# %%
