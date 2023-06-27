#%% import dependencies
import os
import glob
import sys
import numpy as np
import torch
from tqdm import tqdm

from torch import nn, save, load
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import MultiStepLR, CosineAnnealingWarmRestarts, LambdaLR
from torch.optim import Adam
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt


torch.manual_seed(42)
current_directory = os.getcwd()  # Get the current working directory
parent_directory = os.path.abspath(os.path.join(current_directory, os.pardir))  # Get the absolute path of the parent directory

device = 'mps'
TRAINDATASET = os.path.abspath(os.path.join(parent_directory, 'Thesis Code/data/FD001/min-max/train'))
TESTDATASET = os.path.abspath(os.path.join(parent_directory, 'Thesis Code/data/FD001/min-max/test'))
BATCHSIZE = 10
EPOCHS = 100
TRAIN = True


#Frequentist neural network class
class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(NeuralNetwork, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.dense_layers = nn.Sequential(
            nn.Linear(hidden_size, 32), 
            nn.Softplus(),
            nn.Linear(32, 16),
            nn.Softplus(),
            nn.Linear(16,1),
            nn.Softplus(),
        )
        
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device) #initial hidden state
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device) #initial cell state
        
        out, _ = self.lstm(x, (h0, c0))
        out = out[:, -1, :]  # Extract the last time step output
        
        out = self.dense_layers(out) #pass through dense layers
        
        return out

    


#%% Training
if __name__ == "__main__":

    #training and validation data
    # %%Create an instance of the custom dataset
    from Data_loader import CustomDataset
    train = CustomDataset(TRAINDATASET)
    test = CustomDataset(TESTDATASET)
    train_data = DataLoader(train, batch_size=BATCHSIZE)
    test_data = DataLoader(test, batch_size=BATCHSIZE)

    # Model input parameters
    input_size = 14 #number of features
    hidden_size = 32
    num_layers = 1

    #Model, optimizer and loss function
    NNmodel = NeuralNetwork(input_size, hidden_size, num_layers).to(device)
    opt = Adam(NNmodel.parameters(), lr=1e-3)

    # Define the lambda function for decaying the learning rate
    lr_lambda = lambda epoch: (1 - min(60-1, epoch) / 60) ** 0.7
    # Create the learning rate scheduler
    scheduler = LambdaLR(opt, lr_lambda=lr_lambda)

    loss_fn = nn.MSELoss()


    #%% Train the model
    loss_lst = []
    if TRAIN == True:
        for epoch in range(EPOCHS):
            loop = tqdm(train_data)
            for batch in loop:
                X, y = batch
                y = torch.t(y)
                X, y = X.to(device), y.to(device)
                y_pred = NNmodel(X)
                loss = torch.sqrt(loss_fn(y_pred[:,0], y)) #RMSE loss function
                

            
                #Backprop
                opt.zero_grad()
                loss.backward()
                opt.step()
            
                loop.set_description(f"Epoch: {epoch+1}/{EPOCHS}")
                loop.set_postfix(loss = loss.item(), lr = opt.param_groups[0]['lr'])

            scheduler.step() 
            loss_lst.append(loss.item())  

        with open('BNN/model_state.pt', 'wb') as f:
            save(NNmodel.state_dict(), f)

        plt.plot(loss_lst)
        plt.show()

    #%% Test the model
    else:
        with open('BNN/model_state_b5_e10.pt', 'rb') as f: 
            NNmodel.load_state_dict(load(f)) 

        file_paths = glob.glob(os.path.join(TESTDATASET, '*.txt')) 
        file_paths.sort()

        error_lst = []
        loop = tqdm(file_paths)
        for file_path in loop:
        # Process each selected file
            input = np.genfromtxt(file_path, delimiter=" ", dtype=np.float32)
            
            input_tensor = ToTensor()(input).to(device)

            y_pred = NNmodel(input_tensor)
            y = float(file_path[-7:-4])
            error = y_pred.item() - y
            error_lst.append(error)

        RMSE = np.sqrt(np.average(error**2))
        print(f'RMSE = {RMSE}')
    
# %%

