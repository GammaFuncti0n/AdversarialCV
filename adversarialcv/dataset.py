from typing import Dict, Tuple
from torch.utils.data import DataLoader
import os

import torch
import torchvision
from torchvision.datasets import MNIST

class AdversarialDataloader():
    '''
    Class for load dataset and compile it in dataloaders
    '''
    def __init__(self, dataset_config: Dict, train_config: Dict) -> None:
        '''
        Init method AdversarialDataloader
        :params:
            dataset_config: Dict - configuration dictionary with name of dataset and path for it
            train_config: Dict - configuration dictionary with dataloader params
        '''
        self.dataset_config = dataset_config
        self.train_config = train_config

        os.makedirs(self.dataset_config['path'], exist_ok=True)
        if(self.dataset_config['name']=='mnist'):
            self.dataset = MNISTDataset(self.dataset_config['path'])
        else:
            raise Exception(f"Unsupported dataset: {self.dataset_config['name']}")

    def get_dataloaders(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        '''
        Return train/val/test dataloaders
        :returns:
            train_dataloader - dataloader with train data
            val_dataloader - dataloader with valodation data 
            test_dataloader - dataloader with test data
        '''
        if(self.dataset.train_data is not None):
            train_dataloader = torch.utils.data.DataLoader(
                self.dataset.train_data,
                batch_size=self.train_config['batch_size'],
                shuffle=True,
                num_workers=self.train_config['num_workers'],
                pin_memory=True,
                prefetch_factor=self.train_config['num_workers']*2,
                persistent_workers=True
            )
        else:
            train_dataloader = None

        if(self.dataset.val_data is not None):
            val_dataloader = torch.utils.data.DataLoader(
                self.dataset.val_data,
                batch_size=self.train_config['batch_size'],
                shuffle=False,
                num_workers=self.train_config['num_workers'],
                pin_memory=True,
                prefetch_factor=self.train_config['num_workers']*2,
                persistent_workers=True
            )
        else:
            val_dataloader = None

        if(self.dataset.test_data is not None):
            test_dataloader = torch.utils.data.DataLoader(
                self.dataset.test_data,
                batch_size=self.train_config['batch_size'],
                shuffle=False,
                num_workers=self.train_config['num_workers'],
                pin_memory=True,
                prefetch_factor=self.train_config['num_workers']*2,
                persistent_workers=True
            )
        else:
            test_dataloader = None

        return (train_dataloader, val_dataloader, test_dataloader)

class MNISTDataset():
    '''
    MNIST dataset from torchvision dataset
    '''
    def __init__(self, path: str) -> None:
        '''
        Init method MNISTDataset
        :params:
            path - path where load and take dataset
        '''
        self.train_transforms = torchvision.transforms.Compose([
            torchvision.transforms.RandomRotation(30),
            torchvision.transforms.ToTensor(),
        ])  
        self.test_transforms = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
        ])

        self.train_data = MNIST(path, train=True, transform=self.train_transforms, download=True)
        # validation and test the same part of data
        self.val_data = self.test_data = MNIST(path, train=False, transform=self.test_transforms, download=True)