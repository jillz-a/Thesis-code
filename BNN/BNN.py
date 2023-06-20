import os
import sys
import numpy as np
import torch
from torch import nn, save, load
from torch.utils.data import DataLoader, Dataset
from torch.optim import Adam

device = 'mps'
TRAINDATASET = 'data/FD001/min-max/train'
TESTDATASET = 'data/FD001/min-max/test'
BATCHSIZE = 10
EPOCHS = 10

class CustomDataset(Dataset):
    def __init__(self, folder_path):
        self.folder_path = folder_path
        self.file_paths = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith('.txt')]

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        # Load and process the text file data as per your requirement
        data = self.load_data(file_path)
        label = int(file_path[-7:-4])

        # Return the processed data and its corresponding label (if applicable)
        return data, label

    def load_data(self, file_path):
        # Implement the logic to load and process the data from a text file
        # You can use standard file reading techniques or any other processing steps you need
        # Return the processed data as a tensor or in the required format

        # Example: Read data from text file
        with open(file_path, 'r') as file:
            data = file.read()

        # Example: Convert data to tensor
        data_tensor = torch.tensor(data)

        return data_tensor


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
# %%Create an instance of the custom dataset
train = CustomDataset(TRAINDATASET)
test = CustomDataset(TESTDATASET)
train_data = DataLoader(train, batch_size=BATCHSIZE)
test_data = DataLoader(test, batch_size=BATCHSIZE)

#Training
if __name__ == "__main__":
    for epoch in range(EPOCHS):
        for batch in train_data:
            X, y = batch
            X, y = X.to(device), y.to(device)
            y_pred = NNmodel(X)
            loss = loss_fn(y_pred, y)

            #Backprop
            opt.zero_grad()
            loss.backward()
            opt.step()
        
        print(f"Epoch: {epoch} Loss: {loss}")

    with open('BNN/model_state.pt', 'wb') as f:
        save(NNmodel.state_dict(), f)
# %%
