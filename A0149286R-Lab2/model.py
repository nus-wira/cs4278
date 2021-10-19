import torch.nn as nn
import torch.nn.functional as F
import torch
from torch import Tensor
from torchvision import models

alexnet = models.alexnet(pretrained=True)

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class MyModel(nn.Module):

    def __init__(self, num_bins=5):
        super().__init__()
        self.num_bins = num_bins
        # Build the CNN feature extractor
        self.cnn = nn.Sequential(
            *list(alexnet.features.children()),
            alexnet.avgpool
        )
        
        self.fc = nn.Sequential(
            # Add intention to in_features
            nn.Dropout(),
            nn.Linear(in_features=256 * 6 * 6 + 3,out_features=4096),
            *list(alexnet.classifier.children())[2:-1],
            nn.Linear(4096, num_bins)
        )        

        print(f'A simple learner. WARNING: For training efficiency, '
              f'it assumes the intention is the same for all samples in a batch. ')

    def forward(self, image, intention):
        # Map images to feature vectors
        feature = self.cnn(image).flatten(1)
        # Cast intention to one-hot encoding 
        intention = intention.unsqueeze(1)
        onehot_intention = torch.zeros(intention.shape[0], 3, device=intention.device).scatter_(1, intention, 1)
        # Predict control
        control = self.fc(torch.cat((feature, onehot_intention), dim=1)).view(-1, self.num_bins)

        return control
   