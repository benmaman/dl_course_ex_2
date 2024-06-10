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

with open('lfw/pairsDevTrain.txt', 'r') as f:
    train_lines = f.readlines()
with open('lfw/pairsDevTest.txt', 'r') as f:
    test_lines = f.readlines()

# Define transformations
transform = transforms.Compose([
    transforms.Resize((105, 105)),
    transforms.Grayscale(num_output_channels=1),  # Convert images to grayscale
    transforms.ToTensor(),
    transforms.Normalize(mean=[0], std=[1])
])


train_pairs = parse_data(train_lines)
test_pairs = parse_data(test_lines)

# Initialize dataset
train_dataset = FacePairsDataset(train_pairs, "lfw/" ,transform=transform)
test_dataset = FacePairsDataset(test_pairs, 'lfw/', transform=transform)

# DataLoader
data_loader_train = DataLoader(train_dataset, batch_size=100, shuffle=True)
data_loader_test = DataLoader(test_dataset, batch_size=100, shuffle=True)

##------------------------Model train---------------------------------------##

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SiameseNetwork()
train_model(model, device, data_loader_train,data_loader_test, epochs=100,lr=0.0005)  # Assuming data_loader_train is defined