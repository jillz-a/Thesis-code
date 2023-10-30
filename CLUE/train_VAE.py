#%% import dependencies
import glob
import sys
import os
import csv
import json

# Get the absolute path of the project directory
project_path = os.path.dirname(os.path.abspath(os.path.join((__file__), os.pardir)))
# Add the project directory to sys.path if it's not already present
if project_path not in sys.path:
    sys.path.append(project_path)

import numpy as np
import torch
from tqdm import tqdm
from sklearn.model_selection import KFold
import pandas as pd

from torch import nn, save, load
from torch.utils.data import DataLoader, ConcatDataset, SequentialSampler, random_split
from torch.optim.lr_scheduler import LambdaLR
from torch.optim import Adam
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt

from bayesian_torch.models.dnn_to_bnn import get_kl_loss
import bayesian_torch.layers as bl


from variables import *

TRAINDATASET = f'Thesis Code/data/{DATASET}/min-max/train'
TESTDATASET = f'Thesis Code/data/{DATASET}/min-max/test'

