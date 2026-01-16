import torch
import torch.nn as nn
import torchvision
import os

class SimpleCNN(nn.Module):
    '''
    Model with few convolutional layers and 1-Linear head
    '''
    def __init__(self, num_channels, hidden_state, num_classes) -> None:
        '''
        Init method SimpleCNN
        :params:
            num_channels - number of channels for input image, 1 for gray scale and 3 for rgb image
            hidden_state - size of 1-st dim in Linear layer
            num_classes - size of output (number of classes)
        '''
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(num_channels, 16, 3),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3),
            nn.MaxPool2d(2),
            nn.Flatten()
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_state, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        Forward pass SimpleCNN
        '''
        x_ = self.feature_extractor(x)
        out = self.classifier(x_)
        return out

class ResNet18(nn.Module):
    '''
    Model with pretrained ResNet18 backbone
    '''
    def __init__(self, num_classes) -> None:
        '''
        Init method ResNet18
        :params:
            num_classes - size of output (number of classes)
        '''
        super().__init__()

        resnet18_backbone = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
        hidden_state = resnet18_backbone.fc.in_features
        resnet18_backbone = torch.nn.Sequential(*list(resnet18_backbone.children())[:-1])

        self.feature_extractor = resnet18_backbone#.requires_grad_(False)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(hidden_state, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        Forward pass ResNet18
        '''
        if x.size(1)==1:
            x = x.repeat(1,3,1,1)
        x_ = self.feature_extractor(x)
        out = self.classifier(x_)
        return out
