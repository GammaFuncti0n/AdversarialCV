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
    Train runner for fit model on data
    '''
    def __init__(self, config: Dict) -> None:
        '''
        Init method TrainRunner
        :params:
            config: Dict - configuration file with parameters for fitting
        '''
        self.config = config
        self.model_config = config['model']
        self.dataset_config = config['dataset']
    
    def run(self) -> None:
        '''
        Method for run model training
        Initialize dataloaders and fit the model
        '''
        dataloader = AdversarialDataloader(self.dataset_config, self.model_config['params']['training_params'])
        train_dataloader, val_dataloader, test_dataloader = dataloader.get_dataloaders()
        
        model = Module(self.config)
        #model.fit(train_dataloader, val_dataloader)

        model.load_checkpoint()
        score = model.score(test_dataloader)
        logging.info(score)