import torch
import pandas as pd
from PIL import Image
import torchvision.transforms as transforms


class StartingDataset(torch.utils.data.Dataset): 
    """
    Dataset that contains 100000 3x224x224 black images (all zeros).
    """

    def __init__(self):
        file_path = 'data/train-balanced.csv'
        file_path_kaggle = "/kaggle/input/cassava-leaf-disease-classification/train-balanced.csv"
        temp = pd.read_csv(file_path, nrows = 5000) # previously 20480
        self.image_id = temp.image_id
        self.label = temp.label
        self.len = len(temp)
        pass

    def __getitem__(self, index):
        # inputs = torch.zeros([3, 224, 224])
        # label = 0
        file_path = 'data/train_images/'
        file_path_kaggle = "/kaggle/input/cassava-leaf-disease-classification/train_images/"
        image = Image.open(file_path + self.image_id[index])
        reduceSize = transforms.Compose([transforms.Resize((75,100))])
        inputs = reduceSize(transforms.ToTensor()(image))
        label = self.label[index]

        return inputs, label

    def __len__(self):
        return self.len
