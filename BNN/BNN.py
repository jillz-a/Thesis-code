#%% import dependencies
import os
import sys
import numpy as np
import torch
from torch import nn, save, load
from torch.utils.data import DataLoader, Dataset
from torch.optim import Adam
from torchvision.transforms import ToTensor



current_directory = os.getcwd()  # Get the current working directory
parent_directory = os.path.abspath(os.path.join(current_directory, os.pardir))  # Get the absolute path of the parent directory


device = 'mps'
TRAINDATASET = os.path.join(parent_directory, 'Thesis Code/data/FD001/min-max/train')
TESTDATASET = os.path.join(parent_directory, 'Thesis Code/data/FD001/min-max/test')
BATCHSIZE = 10
EPOCHS = 10
TRAIN = True


#%%Frequentist neural network class
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(30*14, 20),
            nn.ReLU(),
            nn.Linear(20,5),
            nn.ReLU(),
            nn.Linear(5, 1),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
    


#Training
if __name__ == "__main__":

    #Model, optimizer and loss function
    NNmodel = NeuralNetwork().to(device)
    opt = Adam(NNmodel.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    #training and validation data
    # %%Create an instance of the custom dataset
    from Data_loader import CustomDataset
    train = CustomDataset(TRAINDATASET)
    test = CustomDataset(TESTDATASET)
    train_data = DataLoader(train, batch_size=BATCHSIZE)
    test_data = DataLoader(test, batch_size=BATCHSIZE)

    if TRAIN == True:
        for epoch in range(EPOCHS):
            for batch in train_data:
                X, y = batch
                X, y = X.to(device), y.to(device)
                y_pred = NNmodel(X)
                loss = loss_fn(y_pred[0][0], y[0])

                #Backprop
                opt.zero_grad()
                loss.backward()
                opt.step()
            
            print(f"Epoch: {epoch} Loss: {loss.item()}")

        with open('BNN/model_state.pt', 'wb') as f:
            save(NNmodel.state_dict(), f)

    else:
        with open('BNN/model_state.pt', 'rb') as f: 
            NNmodel.load_state_dict(load(f)) 
        
        input = np.genfromtxt('/Users/jillesandringa/Documents/AE/MSc/Thesis/Thesis code/data/FD001/min-max/train/train_159-003.txt', delimiter=" ", dtype=np.float32)
        input_tensor = ToTensor()(input).unsqueeze(0).to(device)

        print(NNmodel(input_tensor))
# %%

