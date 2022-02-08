import torch
import torch.nn as nn
import torch.nn.functional as F
import constants


class StartingNetwork(torch.nn.Module):
    """
    Basic logistic regression on 224x224x3 images.
    """
    # what number of channels should we start with? 3?
    # what difference does the kernel size make?

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5, 2)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5, 2)

        self.flatten = nn.Flatten()
        #self.fc = nn.Linear(16 * 72 * 97, 32)
        self.fc = nn.Linear(16 * 3 * 5, constants.BATCH_SIZE)

        # self.fc = nn.Linear(224 * 224 * 3, 1)
        # constants.batch_size
        # putting sigmoid at the end messes everything up
        # everything will be in the range 0 - 1
        # pytorch has a softmax built into the loss calculation
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        #(32, 3, 75, 100)
        #first conv: (32, 6, 75, 100). How does max pooling of (2,2) affect dimensions? 
        x = self.pool(F.relu(self.conv1(x)))
        #second conv:
        x = self.pool(F.relu(self.conv2(x)))
        #second pool:  --> we get dimensions to feed to fc1 here

        # x = self.conv1(x)
        # print(x.shape)
        # x = self.pool(x)
        # print(x.shape)
        # x = F.relu(x)
        # print(x.shape)
        # x = self.conv2(x)
        # print(x.shape)
        # x = self.pool(x)
        # print(x.shape)
        # x = F.relu(x)
        # print(x.shape)

        x = self.flatten(x)
        # print(x.shape)
        
        x = self.fc(x)
        x = self.sigmoid(x)
        return x
