import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from tools import *
from model import *
##---------------------------data processing---------------------------------##

with open('data/pairsDevTrain.txt', 'r') as f:
    train_lines = f.readlines()
with open('data/pairsDevTest.txt', 'r') as f:
    test_lines = f.readlines()

# Define transformations
transform = transforms.Compose([
    transforms.Resize((105, 105)),
    transforms.Grayscale(num_output_channels=1),  # Convert images to grayscale
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485], std=[0.229])  # Adjust mean and std for single channel
])


train_pairs = parse_data(train_lines)
test_pairs = parse_data(test_lines)

# Initialize dataset
train_dataset = FacePairsDataset(train_pairs, 'data/', transform=transform)
test_dataset = FacePairsDataset(test_pairs, 'data/', transform=transform)

# DataLoader
data_loader_train = DataLoader(train_dataset, batch_size=64, shuffle=True)
data_loader_test = DataLoader(test_dataset, batch_size=64, shuffle=True)

##------------------------Model train---------------------------------------##

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SiameseNetwork()
train_model(model, device, data_loader_train,data_loader_test, epochs=100,lr=0.001)  # Assuming data_loader_train is defined