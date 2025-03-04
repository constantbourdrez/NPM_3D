
#
#
#      0===========================================================0
#      |       TP6 PointNet for point cloud classification         |
#      0===========================================================0
#
#
# ------------------------------------------------------------------------------------------
#
#      Jean-Emmanuel DESCHAUD - 21/02/2023
#

import numpy as np
import random
import math
import os
import time
import torch
import scipy.spatial.distance
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torch.nn as nn
import torch.nn.functional as F
import sys
import matplotlib.pyplot as plt
# Import functions to read and write ply files
from ply import write_ply, read_ply
from data_augmentation import PointCloudDataAugmented, default_transforms,  custom_transforms, test_transforms

class MLP(nn.Module):
    def __init__(self, classes = 10):
        super().__init__()
        self.n_classes = classes
        self.layer1 = nn.Linear(3072,512)
        self.layer2 = nn.Linear(512, 256)
        self.layer3 = nn.Linear(256, self.n_classes)
        self.drop = nn.Dropout(p=0.3)
        self.batch_norm = nn.BatchNorm1d(512)
        self.batch_norm2 = nn.BatchNorm1d(256)

    def forward(self, input):
        flattened_input = torch.flatten(input, start_dim=1)
        x = F.relu(self.layer1(flattened_input))
        x = self.batch_norm(x)
        x = F.relu(self.layer2(x))
        x = self.batch_norm2(x)
        x = self.drop(x)
        x = self.layer3(x)
        return x



class PointNetBasic(nn.Module):
    def __init__(self, classes = 10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(3, 64, 1),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Conv1d(64, 64, 1),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Conv1d(64, 64, 1),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Conv1d(64, 128, 1),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Conv1d(128, 1024, 1),
            nn.ReLU(),
            nn.BatchNorm1d(1024),
            nn.MaxPool1d(1024),
            nn.Flatten(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),
            nn.Linear(256, classes),
            nn.Softmax(dim=-1),
        )


    def forward(self, input):
        x = self.net(input)
        return x



class Tnet(nn.Module):
    def __init__(self, k=3):
        super().__init__()
        self.k = k
        self.tnet = nn.Sequential(
            nn.Conv1d(k, 64, 1),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Conv1d(64, 128, 1),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Conv1d(128, 1024, 1),
            nn.ReLU(),
            nn.BatchNorm1d(1024),
            nn.MaxPool1d(1024),
            nn.Flatten(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Linear(256, k*k)
        )


    def forward(self, input):
        x = self.tnet(input)
        return x.view(-1, self.k, self.k)


class PointNetFull(nn.Module):
    def __init__(self, classes = 10):
        super().__init__()
        self.n_classes = classes
        self.tnet1 = Tnet()
        self.net = nn.Sequential(
            nn.Conv1d(3, 64, 1),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Conv1d(64, 64, 1),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Conv1d(64, 64, 1),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Conv1d(64, 128, 1),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Conv1d(128, 1024, 1),
            nn.ReLU(),
            nn.BatchNorm1d(1024),
            nn.MaxPool1d(1024),
            nn.Flatten(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),
            nn.Linear(256, self.n_classes),
        )

    def forward(self, input):
        t_net_transform = self.tnet1(input) + torch.eye(3).to(input.device)
        return self.net(t_net_transform @ input), t_net_transform

def basic_loss(outputs, labels):
    criterion = torch.nn.CrossEntropyLoss()
    return criterion(outputs, labels)

def pointnet_full_loss(outputs, labels, m3x3, alpha = 0.001):
    criterion = torch.nn.CrossEntropyLoss()
    bs=outputs.size(0)
    id3x3 = torch.eye(3, requires_grad=True).repeat(bs,1,1).to(outputs.device)
    if outputs.is_cuda:
        id3x3=id3x3.cuda()
    diff3x3 = id3x3-torch.bmm(m3x3,m3x3.transpose(1,2))
    return criterion(outputs, labels) + alpha * (torch.norm(diff3x3)) / float(bs)

def train(model, device, train_loader, test_loader=None, epochs=250):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    loss=0
    train_loss = []
    test_accuracy = []
    for epoch in range(epochs):
        model.train()
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data['pointcloud'].to(torch.float32).to(device), data['category'].to(device)
            optimizer.zero_grad()
            outputs = model(inputs.transpose(1,2))
            #outputs, m3x3 = model(inputs.transpose(1,2))
            loss = basic_loss(outputs, labels)
            #loss = pointnet_full_loss(outputs, labels, m3x3)
            train_loss.append(loss.item())
            loss.backward()
            optimizer.step()
        scheduler.step()

        model.eval()
        correct = total = 0
        test_acc = 0
        if test_loader:
            with torch.no_grad():
                for data in test_loader:
                    inputs, labels = data['pointcloud'].to(torch.float32).to(device), data['category'].to(device)
                    outputs = model(inputs.transpose(1,2))
                    #outputs, __ = model(inputs.transpose(1,2))
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            test_acc = 100. * correct / total
            test_accuracy.append(test_acc)
            print('Epoch: %d, Loss: %.3f, Test accuracy: %.1f %%' %(epoch+1, loss, test_acc))
    plt.subplots(1, 2, figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(train_loss)
    plt.title('Training loss')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.grid()
    plt.subplot(1, 2, 2)
    plt.plot(test_accuracy)
    plt.title('Test accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.grid()
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    data_aug = True
    t0 = time.time()

    ROOT_DIR = "../data/ModelNet10_PLY"

    device = torch.device("mps")
    print("Device: ", device)
    if data_aug:
        train_ds = PointCloudDataAugmented(ROOT_DIR, folder='train', transform= custom_transforms())
    else:
        train_ds = PointCloudDataAugmented(ROOT_DIR, folder='train', transform=default_transforms())
    test_ds = PointCloudDataAugmented(ROOT_DIR, folder='test', transform=test_transforms())

    inv_classes = {i: cat for cat, i in train_ds.classes.items()}
    print("Classes: ", inv_classes)
    print('Train dataset size: ', len(train_ds))
    print('Test dataset size: ', len(test_ds))
    print('Number of classes: ', len(train_ds.classes))
    print('Sample pointcloud shape: ', train_ds[0]['pointcloud'].size())

    train_loader = DataLoader(dataset=train_ds, batch_size=32, shuffle=True)
    test_loader = DataLoader(dataset=test_ds, batch_size=32)

    #model = MLP()
    model = PointNetBasic()
    #model = PointNetFull()

    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    print("Number of parameters in the Neural Networks: ", sum([np.prod(p.size()) for p in model_parameters]))
    model.to(device);

    train(model, device, train_loader, test_loader, epochs = 150)

    t1 = time.time()
    print("Total time for training : ", t1-t0)
