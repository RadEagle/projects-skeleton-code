import torch
import torch.nn as nn
import torch.nn.functional as F
import constants
import torchvision.transforms as transforms
import torchvision

class StartingNetwork(torch.nn.Module):
    """
    Basic logistic regression on 224x224x3 images.
    """
    # what number of channels should we start with? 3?
    # what difference does the kernel size make?

    def __init__(self):
        super().__init__()

        model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet101', pretrained=True)
        self.newmodel = torch.nn.Sequential(*(list(model.children())[:-1]))
        self.resnetfc = nn.Linear(2048, 5)



        #Add padding, batch normalization
        self.conv1 = nn.Conv2d(3, 6, 5, 2, padding=1)
        self.bn1 = nn.BatchNorm2d(6)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5, 2)
        self.bn2 = nn.BatchNorm2d(16)
        self.dropout = nn.Dropout(0.225)
        self.flatten = nn.Flatten()
        #self.fc = nn.Linear(16 * 72 * 97, 32)

        # for now batch size is fixed, we will reduce our training size
        #self.fc = nn.Linear(16 * 3 * 5, constants.BATCH_SIZE)
        self.fc = nn.Linear(16 * 3 * 5, 16 * 3)
        self.fc2 = nn.Linear(16 * 3, 16)
        self.fc3 = nn.Linear(16, 5)
        #fc2 = nn.Linear(128, 3*5 ) 
        #fc3 = nn.Linear(128, 5 )   #(batchsize, 16*3*5)     --> ...... --> (batchsize, 5)
        #self.normalize = transforms.Normalize(mean=x.mean(), std=x.std())

        # self.fc = nn.Linear(224 * 224 * 3, 1)
        
        self.softmax = nn.Softmax()

        self.deepconv1 = nn.Conv2d(3, 64, 3, 1, padding=1)
        self.deepconv2 = nn.Conv2d(64, 64, 3, 1, padding=1)

        self.bn_conv = nn.BatchNorm2d(64)
        self.bn_affine = nn.BatchNorm2d(500)

        self.deepfc1 = nn.Linear(64 * 56 * 56, 500)
        self.deepfc2 = nn.Linear(500, 500)
        self.deepfc3 = nn.Linear(500, 5)

    def forward(self, x):
        #(32, 3, 75, 100)
        #first conv: (32, 6, 75, 100). How does max pooling of (2,2) affect dimensions?
        # print(x.shape)
        x = self.newmodel(x)
        # print(x.shape)
        x = self.flatten(x)
        # print(x.shape)
        x = self.resnetfc(x)
        # print(x.shape)
         
        # x = F.relu(self.bn1(self.conv1(x)))
        # x = self.pool(x)
        # # print(x.shape)
        # #second conv:
        # x = F.relu(self.bn2(self.conv2(x)))
        # x = self.pool(x)
        # # print(x.shape)
        # #second pool:  --> we get dimensions to feed to fc1 here

        # # x = self.conv1(x)
        # # print(x.shape)
        # # x = self.pool(x)
        # # print(x.shape)
        # # x = F.relu(x)
        # # print(x.shape)
        # # x = self.conv2(x)
        # # print(x.shape)
        # # x = self.pool(x)
        # # print(x.shape)
        # # x = F.relu(x)
        # # print(x.shape)

        # x = self.flatten(x)
        # # print(x.shape)
        
        # x = F.relu(self.fc(x))
        # x = F.relu(self.fc2(x))
        # x = (self.fc3(x))
        # print(f'final shape: {x.shape}')
        # x = self.softmax(x)

        # New model
        # x = F.relu(self.bn_conv(self.deepconv1(x)))
        # x = F.relu(self.bn_conv(self.deepconv2(x)))
        # x = self.pool(x)

        # print(x.shape)

        # x = F.relu(self.bn_conv(self.deepconv2(x)))
        # x = F.relu(self.bn_conv(self.deepconv2(x)))
        # x = self.pool(x)

        # print(x.shape)

        # x = self.flatten(x)

        # print(x.shape)

        # x = F.relu(self.bn_affine(self.deepfc1(x)))
        # x = F.relu(self.bn_affine(self.deepfc2(x)))
        # x = self.deepfc3(x)

        return x
