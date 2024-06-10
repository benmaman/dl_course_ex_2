import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time 
from sklearn.metrics import roc_auc_score



class ContrastiveLoss(torch.nn.Module):

    def __init__(self, margin=0.8):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, input1, input2, y):
        if input1.dim() == 1:
            input1 = input1.unsqueeze(1)
        if input2.dim() == 1:
            input2 = input2.unsqueeze(1)
        diff = input1 - input2
        dist_sq = torch.sum(torch.pow(diff, 2), 1)
        dist = torch.sqrt(dist_sq)
        mdist = self.margin - dist
        dist = torch.clamp(mdist, min=0.0)
        loss = y * dist_sq + (1 - y) * torch.pow(dist, 2)
        loss = torch.sum(loss) / 2.0 / input1.size()[0]
        return loss,dist


class SiameseNetwork(nn.Module):

    """Object of Deep learnnig Siames model
    """
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=10, stride=1, padding=1),  # Assuming input is 1x256x256
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),  # Output size: 64x124x124

            nn.Conv2d(64, 128, kernel_size=7, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),  # Output size: 128x60x60

            nn.Conv2d(128, 256, kernel_size=4, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),  # Output size: 256x30x30

            nn.Conv2d(256, 128, kernel_size=3, padding=1, stride=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2, stride=2),  # Output size: 512*5*5

            nn.Flatten(),  # Flattened output size: 512*15*15 = 115200

            nn.Linear(3200, 1024),  # Correct size
            nn.ReLU(inplace=True),
        )
    def forward_once(self, x):
        return self.conv(x)

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        distance = F.pairwise_distance(output1, output2, keepdim=True)
        return distance

def evaluate_model(model, device, data_loader, criterion):
    model.eval()
    total_loss = 0
    correct = 0
    total_samples = 0
    predictions = []
    true_labels = []
    with torch.no_grad():
        for data1, data2, labels in data_loader:
            data1, data2 = data1.to(device), data2.to(device)
            labels = labels.to(device).float()  # Ensure labels are on the right device and have the correct type

            # Calculate the model outputs
            output1, output2 = model.forward_once(data1), model.forward_once(data2)
            
            # Calculate probabilities from distances for the loss function
            probabilities = torch.sigmoid(F.pairwise_distance(output1, output2))
            
            # Calculate loss using the custom loss function
            loss,probabilities = criterion(output1, output2, labels)  # Pass the model for regularization
            total_loss += loss.item()

            # Calculate auc
            true_labels.extend(labels.detach().numpy())
            predictions.extend(probabilities.detach().numpy())

    average_loss = total_loss / len(data_loader)
    auc_score = roc_auc_score(true_labels, predictions)

    return average_loss, auc_score




def train_model(model, device, train_loader,test_loader, epochs,lr=0.0001, lambda_reg=0.01):
    model = SiameseNetwork()

    # Loss and Optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = ContrastiveLoss()

    #store the results
    train_losses = []
    test_losses = []
    train_auc_results = []
    test_auc_results = []
    best_auc = 0.0
    best_model_path = 'best_siamese_model.pth'
    # Example training loop
    model.train()

    for epoch in range(epochs):
        running_loss = 0.0  
        correct = 0
        total = 0
        start_time = time.time()
        predictions_train=[]
        true_labels_train=[]
        for batch_idx, (data1, data2, labels) in enumerate(train_loader):
            data1, data2, labels = data1.to(device), data2.to(device), labels.to(device)
            labels = labels.squeeze()  # Add an extra dimension to match the output
            optimizer.zero_grad()
            output1, output2 = model.forward_once(data1), model.forward_once(data2)
            probabilities = torch.sigmoid(F.pairwise_distance(output1, output2))
            loss,probabilities = criterion( output1, output2, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            predictions_train.extend(probabilities.detach().numpy())
            true_labels_train.extend(labels.detach().numpy())

            if batch_idx % 100 == 0:
                current_loss = running_loss / (batch_idx + 1)
                auc_score = roc_auc_score(true_labels_train, predictions_train)
                print(f'Train Epoch: {epoch+1} [{batch_idx * len(data1)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}, Train AUC: {auc_score:.2f}')



        epoch_loss = running_loss / len(train_loader)
        auc_score = roc_auc_score(true_labels_train, predictions_train)
        train_losses.append(epoch_loss)
        train_auc_results.append(auc_score)

        test_loss, test_auc = evaluate_model(model, device, test_loader, criterion)
        test_losses.append(test_loss)
        test_auc_results.append(test_auc)

        elapsed_time = time.time() - start_time
        print(f'Epoch {epoch+1}/{epochs}, Time: {elapsed_time:.2f}s, Train Loss: {epoch_loss:.4f}, Train AUC: {auc_score:.2f}, Test Loss: {test_loss:.4f}, Test AUC: {test_auc:.2f}')

        if test_auc > best_auc:
            best_auc = test_auc
            torch.save(model.state_dict(), best_model_path)
            print(f'Saved new best model with accuracy: {best_auc:.2f}%')

    return train_losses, test_losses