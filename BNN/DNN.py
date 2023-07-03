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


torch.manual_seed(42)
current_directory = os.getcwd()  # Get the current working directory
parent_directory = os.path.abspath(os.path.join(current_directory, os.pardir))  # Get the absolute path of the parent directory

device = 'cpu'
DATASET = 'FD001'
TRAINDATASET = os.path.abspath(os.path.join(parent_directory, f'Thesis Code/data/{DATASET}/min-max/train'))
TESTDATASET = os.path.abspath(os.path.join(parent_directory, f'Thesis Code/data/{DATASET}/min-max/test'))
BATCHSIZE = 100
EPOCHS = 10
TRAIN = False


#Frequentist neural network class
class NeuralNetwork(nn.Module):
    def __init__(self, input_size=14, hidden_size=32, num_layers=1):
        super(NeuralNetwork, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.dense_layers = nn.Sequential(
            nn.Linear(hidden_size, 16), 
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
    hidden_size = 32 #numer of hidden cells in LSTM
    num_layers = 1 #number of LSTM layers

    #Model, optimizer and loss function
    NNmodel = NeuralNetwork(input_size, hidden_size, num_layers).to(device)
    opt = Adam(NNmodel.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    # Define the lambda function for decaying the learning rate
    lr_lambda = lambda epoch: (1 - min(60-1, epoch) / 60) ** 0.7
    # Create the learning rate scheduler
    scheduler = LambdaLR(opt, lr_lambda=lr_lambda)

    #%% Train the model
    loss_lst = []
    if TRAIN == True:
        print(f"Training model: {DATASET}")
        print(f"Batch size: {BATCHSIZE}, Epochs: {EPOCHS}")
        for epoch in range(EPOCHS):
            loop = tqdm(train_data)
            for batch in loop:
                X, y = batch #Input sample, true RUL
                y = torch.t(y) #Transpose to fit X dimension
                X, y = X.to(device), y.to(device) #send to device
                y_pred = NNmodel(X) #Run model
                loss = torch.sqrt(loss_fn(y_pred[:,0], y)) #RMSE loss function
                # print(y_pred, y_pred[:,0],y, loss)
                

            
                #Backprop
                opt.zero_grad()
                loss.backward()
                opt.step()
            
                loop.set_description(f"Epoch: {epoch+1}/{EPOCHS}")
                loop.set_postfix(loss = np.sqrt(loss.item()), lr = opt.param_groups[0]['lr'])

            scheduler.step() 
            loss_lst.append(loss.item())  

        with open(f'BNN/model_state_{DATASET}_test.pt', 'wb') as f:
            save(NNmodel.state_dict(), f)

        plt.plot(loss_lst)
        plt.show()

    #%% Test the model
    else:
        model = f'BNN/model_state_{DATASET}_test.pt'
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
            y = float(file_path[-7:-4]) #True value in file name

            if y > 0: #avoid division by 0
                error = (y_pred.item() - y)/y * 100

            error_lst.append(error)
            y_lst.append(y)
            y_pred_lst.append(y_pred.item())

        RMSE = np.sqrt(np.average([i**2 for i in error_lst])) #Root Mean Squared Error
        print(f'RMSE = {RMSE}')

        y_lst, y_pred_lst = (list(t) for t in zip(*sorted(zip(y_lst, y_pred_lst), reverse=True, key=lambda x: x[0])))
        
        plt.plot(y_lst, label='True RUL')
        plt.scatter([i for i in range(len(y_pred_lst))], y_pred_lst, label='Predicted RUL', c='red')
        plt.xlabel('Engine (test)')
        plt.ylabel('RUL')
        plt.title(f'RUL prediction for {DATASET}. RMSE = {np.round(RMSE, 2)}')
        plt.legend()
        plt.show()
        
    
# %%

