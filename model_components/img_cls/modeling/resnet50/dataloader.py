import os
import pandas as pd
import torch
# from torchvision.io import read_image
from torchvision import transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision
import cv2



class ImageDataset():
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        #total = len(self.img_labels)
        # total = 500
        #return (total - total%32)
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = cv2.imread(img_path)
        if not os.path.exists(img_path):
            print(img_path)
        image = image.reshape((3, 256, 256))
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label

train_dataset = ImageDataset("./data/train.csv","./data")
train_loader=DataLoader(dataset=train_dataset,batch_size=32, shuffle=True)

test_dataset=ImageDataset("./data/test.csv","./data")
test_loader=DataLoader(dataset=test_dataset,batch_size=32)
