#%% import dependencies
import os
import glob
import sys
import numpy as np
import torch
from tqdm import tqdm
from sklearn.model_selection import KFold

from torch import nn, save, load
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler
from torch.optim.lr_scheduler import LambdaLR
from torch.optim import Adam
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt

from bayesian_torch.models.dnn_to_bnn import dnn_to_bnn, get_kl_loss
import bayesian_torch.layers as bl
from bayesian_torch.utils.util import predictive_entropy, mutual_information

torch.manual_seed(42)
current_directory = os.getcwd()  # Get the current working directory
parent_directory = os.path.abspath(os.path.join(current_directory, os.pardir))  # Get the absolute path of the parent directory

device = 'cpu'
DATASET = 'FD001'
TRAINDATASET = os.path.abspath(os.path.join(parent_directory, f'Thesis Code/data/{DATASET}/min-max/train'))
TESTDATASET = os.path.abspath(os.path.join(parent_directory, f'Thesis Code/data/{DATASET}/min-max/test'))
BATCHSIZE = 50
EPOCHS = 20
k = 10 #amount of folds for cross validation

TRAIN = True
CV = False #Cross validation, if Train = True and CV = False, the model will train on the entire data-set

#Bayesian neural network class
class BayesianNeuralNetwork(nn.Module):
    """Bayesian Neural Network using LSTM and linear layers. Deterministic to Bayesian using Reparameterization.

    Args:
        nn (_type_): _description_
    """
    def __init__(self, input_size=14, hidden_size=32, num_layers=1, prior_mean = 0.0, prior_variance = 1.0, posteror_mu_init = 0.0, posterior_rho_init = -3.0):
        super(BayesianNeuralNetwork, self).__init__()
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
       
        out = self.l2(out[0])
    
        return out

#Training loop per epoch
def train_epoch(train_data):
    loop = tqdm(train_data)
    loss_lst = []
    
    for batch in loop:
        X, y = batch #Input sample, true RUL
        y = torch.t(y) #Transpose to fit X dimension
       
        X, y = X.to(device), y.to(device) #send to device
        y_pred = BNNmodel(X) #Run model

        kl = get_kl_loss(BNNmodel) #Kullback Leibler loss
        ce_loss = torch.sqrt(loss_fn(y_pred[0][:,0], y)) #RMSE loss function
        loss = ce_loss + kl / BATCHSIZE
        loss_lst.append(loss.item())
        # print(y_pred, y_pred[:,0],y, loss)
        
    
        #Backprop
        opt.zero_grad()
        loss.backward()
        opt.step()
    
        train_loss = np.average(loss_lst)
        loop.set_description(f"Epoch: {epoch+1}/{EPOCHS}")
        loop.set_postfix(loss = train_loss, lr = opt.param_groups[0]['lr']) 
    
    return train_loss

#Validation loop per epoch
def val_epoch(val_data):
    loop = tqdm(val_data)
    loss_lst = []
    for batch in loop:
        
        X, y = batch #Input sample, true RUL
        y = torch.t(y) #Transpose to fit X dimension
        X, y = X.to(device), y.to(device) #send to device
        y_pred = BNNmodel(X) #Run model
        kl = get_kl_loss(BNNmodel)
        ce_loss = torch.sqrt(loss_fn(y_pred[0][:,0], y)) #RMSE loss function
        loss = ce_loss + kl / BATCHSIZE
        loss_lst.append(loss.item())


        val_loss = np.average(loss_lst)
        loop.set_description(f"Epoch: {epoch+1}/{EPOCHS}")
        loop.set_postfix(loss = val_loss) 
   
    return val_loss


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

    # Import into trained machine learning models
    input_size = 14
    hidden_size = 32
    num_layers = 1

    BNNmodel = BayesianNeuralNetwork(input_size, hidden_size, num_layers).to(device)
    opt = Adam(BNNmodel.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    # Define the lambda function for decaying the learning rate
    lr_lambda = lambda epoch: (1 - min(60-1, epoch) / 60) ** 0.7
    # Create the learning rate scheduler
    scheduler = LambdaLR(opt, lr_lambda=lr_lambda)

    #%% Train the model
    
    if TRAIN == True and CV == False:

        print(f"Training model: {DATASET}")
        print(f"Batch size: {BATCHSIZE}, Epochs: {EPOCHS}")

        train_data = DataLoader(train, batch_size=BATCHSIZE)
        loss_lst = []
        for epoch in range(EPOCHS):

            train_loss = train_epoch(train_data=train_data)
            
            scheduler.step() 
            loss_lst.append(train_loss)  

        with open(f'BNN/BNN_model_state_{DATASET}_test.pt', 'wb') as f:
            save(BNNmodel.state_dict(), f)

        plt.plot(loss_lst)
        plt.show()

    #%% Cross validation
    elif TRAIN == True and CV == True: #Perfrom Cross Validation
        splits = KFold(n_splits=k)
        history = {'train loss': [], 'validation loss': []}

        for fold , (train_idx, val_idx) in enumerate(splits.split(np.arange(len(train)))):
            
            print(f'Fold {fold + 1}')

            train_sampler = SubsetRandomSampler(train_idx)
            test_sampler = SubsetRandomSampler(val_idx)
            train_data = DataLoader(train, batch_size=BATCHSIZE, sampler=train_sampler)
            val_data = DataLoader(train, batch_size=BATCHSIZE, sampler=test_sampler)

            BNNmodel = BayesianNeuralNetwork(input_size, hidden_size, num_layers).to(device)
            opt = Adam(BNNmodel.parameters(), lr=1e-3)

            # Define the lambda function for decaying the learning rate
            lr_lambda = lambda epoch: (1 - min(int(0.6*EPOCHS)-1, epoch) / int(0.6*EPOCHS)) ** 0.7 #after 60% of epochs reach 70% of learning rate
            # Create the learning rate scheduler
            scheduler = LambdaLR(opt, lr_lambda=lr_lambda)

            for epoch in range(EPOCHS):
                train_loss = train_epoch(train_data=train_data)
                val_loss = val_epoch(val_data=val_data)

                history['train loss'].append(train_loss)
                history['validation loss'].append(val_loss)

                scheduler.step()

        print(f'Performance of {k} fold cross validation')
        print(f'Average training loss: {np.mean(history["train loss"])}')
        print(f'Average validation loss: {np.mean(history["validation loss"])}')

    #%% Test the model
    else:
        model = f'BNN/BNN_model_state_{DATASET}.pt'
        print(f"Testing model: {model}")
        #load pre trained model
        with open(model, 'rb') as f: 
            BNNmodel.load_state_dict(load(f)) 

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

            y_pred = BNNmodel(input_tensor) #Run model
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