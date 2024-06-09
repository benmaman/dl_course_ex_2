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


class RegularizedCrossEntropyLoss(nn.Module):
    def __init__(self, lambda_reg=0.01):
        super(RegularizedCrossEntropyLoss, self).__init__()
        self.lambda_reg = lambda_reg
        self.bce_loss = nn.BCELoss()

    def forward(self, outputs, labels, model):
        # Binary cross-entropy loss
        bce_loss = self.bce_loss(outputs, labels)

        # L2 regularization
        l2_reg = sum(param.pow(2).sum() for param in model.parameters())

        # Total loss
        total_loss = bce_loss + self.lambda_reg * l2_reg
        return total_loss


class SiameseNetwork(nn.Module):

    """Object of Deep learnnig Siames model
    """
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=10, stride=1),  # Adjust the input channel as necessary
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(64, 128, kernel_size=7, stride=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(128, 128, kernel_size=4, stride=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(128, 256, kernel_size=4, stride=1),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(256 * 6 * 6, 4096),  # Adjust the size based on the output of the last conv layer
            nn.Sigmoid()
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
    total = 0
    correct = 0
    running_loss = 0

    with torch.no_grad():
        for data1, data2, labels in data_loader:
            data1, data2, labels = data1.to(device), data2.to(device), labels.to(device)
            labels = labels.unsqueeze(1)
            outputs = model(data1, data2)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            predicted = (torch.sigmoid(outputs) > 0.5).float()
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    return running_loss / len(data_loader), 100 * correct / total




def train_model(model, device, train_loader,test_loader, epochs,lr=0.0001, lambda_reg=0.01):
    model = SiameseNetwork()

    # Loss and Optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = RegularizedCrossEntropyLoss(lambda_reg=lambda_reg)

    #store the results
    train_losses = []
    test_losses = []
    train_accuracy = []
    test_accuracy = []
    best_accuracy = 0.0
    best_model_path = 'best_siamese_model.pth'
    # Example training loop
    model.train()

    for epoch in range(epochs):
        running_loss = 0.0  
        correct = 0
        total = 0
        start_time = time.time()
        total_loss=0

        for batch_idx, (data1, data2, labels) in enumerate(train_loader):
            data1, data2, labels = data1.to(device), data2.to(device), labels.to(device)
            labels = labels.squeeze()  # Add an extra dimension to match the output
            optimizer.zero_grad()
            output1, output2 = model.forward_once(data1), model.forward_once(data2)
            probabilities = torch.sigmoid(F.pairwise_distance(output1, output2))
            loss = criterion(probabilities, labels, model)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            print(loss.item())
        print(f'Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(train_loader):.4f}')

        #     #update reuslts
        #     running_loss += loss.item()
        #     predicted = (torch.sigmoid(outputs) > 0.5).float()
        #     correct += (predicted == labels).sum().item()
        #     total += labels.size(0)


        #     if batch_idx % 10 == 0:
        #         print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        #             epoch, batch_idx * len(data1), len(train_loader.dataset),
        #             100. * batch_idx / len(train_loader), loss.item()))
                
        # epoch_loss = running_loss / len(train_loader)
        # epoch_acc = 100 * correct / total
        # train_losses.append(epoch_loss)
        # train_accuracy.append(epoch_acc)

        # test_loss, test_acc = evaluate_model(model, device, test_loader, criterion)
        # test_losses.append(test_loss)
        # test_accuracy.append(test_acc)

        # elapsed_time = time.time() - start_time

        # print(f'Epoch {epoch+1}/{epochs}, Time: {elapsed_time:.2f}s, Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.2f}%, Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%')
        # if test_acc > best_accuracy:
        #     best_accuracy = test_acc
        #     torch.save(model.state_dict(), best_model_path)
        #     print(f'Saved new best model with accuracy: {test_acc:.2f}%')

    # return train_losses, train_accuracy, test_losses, test_accuracy
    return