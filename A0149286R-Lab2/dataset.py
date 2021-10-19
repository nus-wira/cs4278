import os
import cv2
import random
import math
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data import Sampler
from torchvision import transforms
from torchvision.transforms import *
import sys
import torch
import math


angle = math.pi / 18 # 20 degrees
n_size = int(112 * math.cos(angle) / (1 + math.tan(angle)))

def read_image(path):
    return Image.open(path)


class MyDataset(Dataset):
    
    INTENTION_MAPPING = {'forward': 0, 'left': 1, 'right': 2}
    MAX_VELOCITY = 0.7
    MIN_VELOCITY = -0.7

    def __init__(self, is_train=True, num_bins=5):
        self.bin_size = (self.MAX_VELOCITY - self.MIN_VELOCITY) / num_bins
        self.is_train = is_train
        self.data_dir = '.'
        if is_train:
            self.data = pd.read_csv(os.path.join(self.data_dir, 'train.txt'), sep='  ')
            # self.data = pd.read_csv(os.path.join(self.data_dir, 'val.txt'), sep=' ')
        else:
            self.data = pd.read_csv(os.path.join(self.data_dir, 'val.txt'), sep=' ')

        self.preprocess = Compose([
            Resize((112, 112)),
            ToTensor(),
            Normalize(mean=[0.5071, 0.4866, 0.4409], std=[0.2675, 0.2565, 0.2761])
        ])
        
        print(f'loaded data from {self.data_dir}. dataset size {len(self)}')

    def discretize_control(self, control):
        return int((control - self.MIN_VELOCITY) / self.bin_size)

    def __getitem__(self, idx):
        # print(self.data.iloc[idx])
        # sys.stdout.flush()
        frame, _, _, angular_velocity, intention = self.data.iloc[idx]
        image = self.preprocess(read_image(os.path.join(self.data_dir, 'images', f'{frame}.jpg')))
        intention = torch.tensor(self.INTENTION_MAPPING[intention])
        label = torch.tensor(self.discretize_control(angular_velocity))
        if self.is_train:
            image, intention, label = self.rand_transform(image, intention, label)

        return image, intention, label

    def rand_transform(self, image, intention, label):
       image, intention, label = self.rand_flip(image, intention, label)
       image = self.rand_rotate(image, 0.2)
       image = self.rand_crop(image, 0.2)
       image = self.rand_blur(image, 0.2)
       image = RandomPerspective(p=0.2).forward(image)
       image = RandomAutocontrast(p=0.2).forward(image)
       return image, intention, label

    def rand_flip(self, image, intention, label, prob=0.5):
       if random.random() > prob:
          return image, intention, label
       f_intention = 2 - intention
       f_label = 4 - label
       f_image = RandomHorizontalFlip(1).forward(image)
       return f_image, f_intention, f_label

    def rand_crop(self, image, prob=0.5):
       if random.random() > prob:
          return image
       c_image = RandomResizedCrop((112,112),(0.5,1.0)).forward(image)
       return c_image
  
    def rand_rotate(self, image, prob=0.5):
       if random.random() > prob:
          return image
       r_image = RandomRotation(angle).forward(image)
       
       # c_image = functional.center_crop(r_image,(n_size,n_size))
       # r_image = Resize((112,112)).forward(c_image)
       return r_image
        
    def rand_blur(self, image, prob=0.5):
        if random.random() > prob:
            return image
        b_image = GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)).forward(image)
        return b_image


    def __len__(self):
        return len(self.data)