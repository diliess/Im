import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
import random
class Caltech256Dataset(Dataset):
    def __init__(self, root_dir, transforms=None,train=True):
        self.root_dir = root_dir
        self.transforms = transforms
        self.train = train
        self.class_names = self._get_class_names()
        self.data = self._load_data(self.train)
        

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name, label = self.data[idx]

        img_path = os.path.join(self.root_dir, img_name)
        image = Image.open(img_path).convert('RGB')

        if self.transforms:
            image = self.transforms(image)

        return image, label

    def _get_class_names(self):
        class_names = sorted(os.listdir(os.path.join(self.root_dir, 'images')))
        return class_names

    def _load_data(self,train):
        data = []
        if train:
            with open(os.path.join(self.root_dir, 'labels', 'train.txt'), 'r') as file:
                for line in file:
                    img_name, label = line.strip().split()
                    data.append((img_name, int(label)))

            random.shuffle(data)   
            return data
        else:
            with open(os.path.join(self.root_dir, 'labels', 'val.txt'), 'r') as file:
                for line in file:
                    img_name, label = line.strip().split()
                    data.append((img_name, int(label)))
            random.shuffle(data)   
            return data

