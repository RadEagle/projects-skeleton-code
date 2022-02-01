import torch
import torch.nn as nn
import torch.nn.functional as F



class StartingNetwork(torch.nn.Module):
    """
    Basic logistic regression on 224x224x3 images.
    """
    # what number of channels should we start with? 3?
    # what difference does the kernel size make?

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)

        self.flatten = nn.Flatten()
        self.fc = nn.Linear(16 * 72 * 97, 32)

        # self.fc = nn.Linear(224 * 224 * 3, 1)
        # constants.batch_size
        # putting sigmoid at the end messes everything up
        # everything will be in the range 0 - 1
        # pytorch has a softmax built into the loss calculation
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))

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
