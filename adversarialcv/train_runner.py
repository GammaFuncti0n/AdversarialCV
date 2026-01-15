import numpy as np
from tqdm import tqdm
from typing import Dict
import logging

import torch
from torch import nn
import torchvision

from adversarialcv.dataset import AdversarialDataloader
from adversarialcv.module import Module

class TrainRunner():
    '''
    Class for init model and fit on data 
    '''
    def __init__(self, config: Dict):
        '''
        Init method TrainRunner
        '''
        self.config = config # main config
        self.model_config = config['model']
        self.dataset_config = config['dataset']
    
    def run(self) -> None:
        '''
        Method for run model training
        Initialize dataloaders, fit model, if needed validate and save model
        '''
        dataloader = AdversarialDataloader(self.dataset_config, self.model_config['params']['training_params'])
        train_dataloader, val_dataloader, test_dataloader = dataloader.get_dataloaders()
        
        model = Module(self.config)
        model.fit(train_dataloader, val_dataloader)

        ### TO DO ###
        # evaluate on test data