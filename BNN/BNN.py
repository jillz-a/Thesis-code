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
import bayesian_torch.layers as bl
from DNN import NeuralNetwork

torch.manual_seed(42)
current_directory = os.getcwd()  # Get the current working directory
parent_directory = os.path.abspath(os.path.join(current_directory, os.pardir))  # Get the absolute path of the parent directory

#Bayesian neural network class
class NeuralNetwork(nn.Module):
    """Bayesian Neural Network using LSTM and linear layers. Deterministic to Bayesian using Reparameterization.

    Args:
        nn (_type_): _description_
    """
    def __init__(self, input_size=14, hidden_size=32, prior_mean = 0.0, prior_variance = 1.0, posteror_mu_init = 0.0, posterior_rho_init = -3.0):
        super(NeuralNetwork, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = bl.LSTMReparameterization(in_features= input_size, out_features= hidden_size, prior_mean=prior_mean, prior_variance=prior_variance, posterior_mu_init=posteror_mu_init, posterior_rho_init=posterior_rho_init)
        self.relu = bl.ReLU()
        self.l1 = bl.LinearReparameterization(in_features=hidden_size, out_features=16)
        self.l2 = bl.LinearReparameterization(16,1)
        
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device) #initial hidden state
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device) #initial cell state
        
        out = self.lstm(x)#, (h0, c0))
        
        out = out[0][:, -1, :]  # Extract the last time step output
        
        out = self.l1(out) #pass through dense layers
        out = self.l2(out)
        
        return out



device = 'cpu'
DATASET = 'FD001'
TRAINDATASET = os.path.abspath(os.path.join(parent_directory, f'Thesis Code/data/{DATASET}/min-max/train'))
TESTDATASET = os.path.abspath(os.path.join(parent_directory, f'Thesis Code/data/{DATASET}/min-max/test'))
BATCHSIZE = 1
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
        "moped_delta": 0,
}

    # dnn_to_bnn(NNmodel, const_bnn_prior_parameters)
 

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
                print(X[0])
                y_pred = NNmodel(X[0]) #Run model
                kl = get_kl_loss(NNmodel)
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
    #%% Test the model
    else:
        model = f'BNN/BNN_model_state_{DATASET}.pt'
        print(f"Testing model: {model}")
        #load pre trained model
        with open(model, 'rb') as f: 
            NNmodel.load_state_dict(load(f)) 

        file_paths = glob.glob(os.path.join(TESTDATASET, '*.txt')) #all samples to test
        file_paths.sort() #sort in chronological order

        error_lst = [] #difference between true and predicted RUL
        y_lst = [] #True RUL values
        y_pred_lst = [] #Predicted RUL values
        loop = tqdm(file_paths)
        for file_path in loop:
        # Process each selected file
            input = np.genfromtxt(file_path, delimiter=" ", dtype=np.float32)
            input_tensor = ToTensor()(input).to(device)

            y_pred = NNmodel(input_tensor) #Run model
            # print(torch.mean(y_pred[0,:,0]))
            y = float(file_path[-7:-4]) #True value in file name

            if y > 0: #avoid division by 0
                error = (torch.mean(y_pred[0,:,0]).item() - y)/y * 100

            error_lst.append(error)
            y_lst.append(y)
            y_pred_lst.append(torch.mean(y_pred[0,:,0]).item())

        RMSE = np.sqrt(np.average([i**2 for i in error_lst])) #Root Mean Squared Error
        print(f'RMSE = {RMSE}')

        y_lst, y_pred_lst = (list(t) for t in zip(*sorted(zip(y_lst, y_pred_lst), reverse=True, key=lambda x: x[0])))
        
        plt.plot(y_lst, label='True RUL')
        plt.scatter([i for i in range(len(y_pred_lst))], y_pred_lst, label='Predicted RUL', c='red')
        plt.xlabel('Engine (test)')
        plt.ylabel('RUL')
        plt.legend()
        plt.show()