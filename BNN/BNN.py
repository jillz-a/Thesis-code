#%% import dependencies
import os
import glob
import sys
import numpy as np
import torch
from tqdm import tqdm
from sklearn.model_selection import KFold, train_test_split

from torch import nn, save, load
from torch.utils.data import DataLoader, ConcatDataset, SequentialSampler, random_split
from torch.optim.lr_scheduler import LambdaLR
from torch.optim import Adam
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt

from bayesian_torch.models.dnn_to_bnn import dnn_to_bnn, get_kl_loss
import bayesian_torch.layers as bl

from EarlyStopping import  EarlyStopping

from variables import *


torch.manual_seed(42)
current_directory = os.getcwd()  # Get the current working directory
parent_directory = os.path.abspath(os.path.join(current_directory, os.pardir))  # Get the absolute path of the parent directory

TRAINDATASET = os.path.abspath(os.path.join(parent_directory, f'Thesis Code/data/{DATASET}/min-max/train'))
TESTDATASET = os.path.abspath(os.path.join(parent_directory, f'Thesis Code/data/{DATASET}/min-max/test'))

TRAIN = True
CV = False #Cross validation, if Train = True and CV = False, the model will train on the entire data-set

#Bayesian neural network class
class BayesianNeuralNetwork(nn.Module):
    """Bayesian Neural Network using LSTM and linear layers. Deterministic to Bayesian using Reparameterization.

    Args:
        input_size: number of input features
        hidden_szie: size of hidden node vector (also size of output)
        num_layers: amountof LSTM layers
        prior_mean: initial guess for parameter mean
        prior_variance: initial guess for parameter variance
        posterior_mu_init: init std for the trainable mu parameter, sampled from N(0, posterior_mu_init)
        posterior_rho_init: init std for the trainable rho parameter, sampled from N(0, posterior_rho_init)

    """
    def __init__(self, input_size=14, hidden_size=32, num_layers=1, prior_mean = 0.0, prior_variance = 1.0, posterior_mu_init = 0.0, posterior_rho_init = -3.0):
        super(BayesianNeuralNetwork, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = bl.LSTMReparameterization(in_features= input_size, out_features= hidden_size, prior_mean=prior_mean, prior_variance=prior_variance, posterior_mu_init=posterior_mu_init, posterior_rho_init=posterior_rho_init)
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
def train_epoch(train_data, model, loss_fn, opt):
    """Trains the model over one epoch

    Args:
        train_data (array (torch)): Input time series data [30, 14]
        model (torch model): (neural network) pytorch model
        loss_fn (_type_): Loss function
        opt (_type_): Optimizer

    Returns:
        float : RMSE training loss
    """
    model.train()
    loop = tqdm(train_data)
    
    for batch in loop:
        X, y = batch #Input sample, true RUL
        y = torch.t(y) #Transpose to fit X dimension
       
        X, y = X.to(device), y.to(device) #send to device
        y_pred = model(X) #Run model

        kl = get_kl_loss(model) #Kullback Leibler loss
        ce_loss = torch.sqrt(loss_fn(y_pred[0][:,0], y)) #RMSE loss function
        loss = ce_loss + kl / BATCHSIZE #Loss including the KL loss
      
        
    
        #Backprop
        opt.zero_grad()
        loss.backward()
        opt.step()
    
        train_loss = ce_loss.item()

        loop.set_description(f"Epoch: {epoch+1}/{EPOCHS}")
        loop.set_postfix(train_loss = train_loss)#, lr = opt.param_groups[0]['lr']) 

    scheduler.step()
    
    return train_loss

#Validation loop per epoch
def test_epoch(test_data, model, loss_fn):
    """Trains the model over one epoch

    Args:
        train_data (array (torch)): Input time series data [30, 14]
        model (torch model): (neural network) pytorch model
        loss_fn (_type_): Loss function

    Returns:
        float : RMSE validation loss
    """
    loop = tqdm(test_data)
    loss_lst = []
    for batch in loop:
        
        X, y = batch #Input sample, true RUL
        # y = torch.t(y) #Transpose to fit X dimension
        X, y = X.to(device), y.to(device) #send to device

        n_samples = 10

        mc_pred = [model(X)[0] for _ in range(n_samples)]

        predictions = torch.stack(mc_pred)
        mean_pred = torch.mean(predictions, dim=0)

        ce_loss = torch.sqrt(loss_fn(mean_pred[:,0], y)) #RMSE loss function
        loss_lst.append(ce_loss.item())


        test_loss = np.mean(loss_lst)
        loop.set_description(f"Test: {epoch+1}/{EPOCHS}")
        loop.set_postfix(test_loss = test_loss) 
   
    return test_loss


# with open(f'BNN/model_state_{DATASET}.pt', 'rb') as f: 
#     NNmodel.load_state_dict(load(f)) 

#%% main script
if __name__ == '__main__':

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
    lr_lambda = lambda epoch: 1 - (min(int(0.6*EPOCHS), epoch) / int(0.6*EPOCHS)) * (1 - 0.7) #after 60% of epochs reach 70% of learning rate
    # Create the learning rate scheduler
    scheduler = LambdaLR(opt, lr_lambda=lr_lambda)

    #early stopping class that stops training to prevent overfitting
    es = EarlyStopping()

    #%% Train the model
    if TRAIN == True and CV == False:

        print(f"Training model: {DATASET}")
        print(f"Batch size: {BATCHSIZE}, Epochs: {EPOCHS}")

        train_set, val_set = random_split(train, [0.8, 0.2])

        train_data = DataLoader(train_set, batch_size=BATCHSIZE)
        val_data = DataLoader(val_set, batch_size= len(val_set))

        train_loss_lst = []
        val_loss_lst = []
        epoch = 0
        done = False
        while epoch < EPOCHS and not done:
            epoch +=1

            train_loss = train_epoch(train_data=train_data, model=BNNmodel, loss_fn=loss_fn, opt=opt)
            val_loss = test_epoch(test_data=val_data, model=BNNmodel, loss_fn=loss_fn)

            train_loss_lst.append(train_loss) 
            val_loss_lst.append(val_loss) 
            

            if es(model=BNNmodel, val_loss=val_loss): done = True

        with open(f'BNN/BNN_model_state_{DATASET}_test.pt', 'wb') as f:
            save(BNNmodel.state_dict(), f)

        plt.plot(train_loss_lst, label='Train loss')
        plt.plot(val_loss_lst, label='Validation loss')
        plt.legend()
        plt.show()

    #%% Cross validation
    elif CV == True: #Perfrom Cross Validation
        splits = KFold(n_splits=k)
        history = {'Train loss': [], 'Test loss': []}
        total_set = ConcatDataset([train, test]) #for cross validation we look at the entire data set

        for fold , (train_idx, test_idx) in enumerate(splits.split(np.arange(len(total_set)))):
            
            print(f'Fold {fold + 1}')

            #Train and test data split according to amount of folds
            train_sampler = SequentialSampler(train_idx) #(k-1)/k part of the total set
            test_sampler = SequentialSampler(test_idx) #1/k part of the total set
            train_data = DataLoader(total_set, batch_size=BATCHSIZE, sampler=train_sampler)
            test_data = DataLoader(total_set, batch_size= BATCHSIZE, sampler=test_sampler)
          

            BNNmodel = BayesianNeuralNetwork(input_size, hidden_size, num_layers).to(device)
            opt = Adam(BNNmodel.parameters(), lr=1e-3)

            # Create the learning rate scheduler
            scheduler = LambdaLR(opt, lr_lambda=lr_lambda)

            loss_lst = [] #list of train loss per epoch
            for epoch in range(EPOCHS):
                train_loss = train_epoch(train_data=train_data, model=BNNmodel, loss_fn=loss_fn, opt=opt)
                loss_lst.append(train_loss)
                
                scheduler.step()

            test_loss = test_epoch(test_data=test_data, model=BNNmodel, loss_fn=loss_fn)
            history['Test loss'].append(test_loss)
            history['Train loss'].append(np.mean(loss_lst))

        print(f'Performance of {k} fold cross validation')
        print(f'Average training loss: {np.mean(history["Train loss"])}')
        print(f'Average test loss: {np.mean(history["Test loss"])}')

    #%% Test the model
    else:
        model = f'BNN/BNN_model_state_{DATASET}_test.pt'
        print(f"Testing model: {model}")

        #Import into trained machine learning models
        with open(model, 'rb') as f: 
            BNNmodel.load_state_dict(load(f)) 
       

        file_paths = glob.glob(os.path.join(TESTDATASET, '*.txt')) #all samples to test
        file_paths.sort() #sort in chronological order
        # file_paths = sorted(file_paths, key=lambda x: x[-7:-4], reverse=True)
       

        #setup data to plot
        mean_pred_lst = []
        true_lst = []
        var_pred_lst = []


        #%%Go through each sample
        loop = tqdm(file_paths[179:179+184])
        for file_path in loop:
            # Process each selected file
            sample = np.genfromtxt(file_path, delimiter=" ", dtype=np.float32)
            label = float(file_path[-7:-4]) #true RUL

            #predict RUL from samples using Monte Carlo Sampling
            X = ToTensor()(sample).to(device)
            n_samples = 10

            mc_pred = [BNNmodel(X)[0] for _ in range(n_samples)]


            predictions = torch.stack(mc_pred)
            mean_pred = torch.mean(predictions, dim=0)
            var_pred = torch.var(predictions, dim=0)
            y = label #True RUL

            #add predictions and true labels to lists
            mean_pred_lst.append(mean_pred.item())
            true_lst.append(y)
            var_pred_lst.append(var_pred.item())

        
        error = [(mean_pred_lst[i] - true_lst[i])**2 for i in range(len(true_lst))]
        B_RMSE = np.round(np.sqrt(np.mean(error)), 2)

        plt.plot(mean_pred_lst, label= f'Bayesian Mean Predicted RUL values, RMSE = {B_RMSE}')
        plt.plot(true_lst, label='True RUL values')
        plt.fill_between(x=np.arange(len(mean_pred_lst)), 
                        y1= mean_pred_lst + np.sqrt(var_pred_lst), 
                        y2=mean_pred_lst - np.sqrt(var_pred_lst),
                        alpha= 0.5,
                        label= '1 STD interval'
                        )
        plt.fill_between(x=np.arange(len(mean_pred_lst)), 
                        y1= mean_pred_lst + 2*np.sqrt(var_pred_lst), 
                        y2=mean_pred_lst - 2*np.sqrt(var_pred_lst),
                        alpha= 0.3,
                        label= '2 STD interval'
                        )

        plt.xlabel('Cycles')
        plt.ylabel('RUL')
        plt.title(f'Dataset {DATASET}, {n_samples} samples per data point, average variance = {np.round(np.mean(var_pred_lst),2)}')
        plt.legend()
        plt.show()