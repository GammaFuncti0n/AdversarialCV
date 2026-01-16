import numpy as np
from tqdm import tqdm
from typing import Dict
import logging

import torch
from torch import nn
import torchvision

from adversarialcv.dataset import AdversarialDataloader
from adversarialcv.module import Module
from adversarialcv.attacker import FGSMAttack, PGDAttack

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
        model.fit(train_dataloader, val_dataloader)

        model.load_checkpoint()
        score = model.score(test_dataloader)
        logging.info(score)

class EvalRunner():
    '''
    Evaluation runner, load model and evaluate on test data
    '''
    def __init__(self, config: Dict) -> None:
        '''
        Init method EvalRunner
        :params:
            config: Dict - configuration file with parameters
        '''
        self.config = config
        self.model_config = config['model']
        self.dataset_config = config['dataset']
    
    def run(self) -> None:
        '''
        Method for evaluate model
        Initialize dataloaders and evaluate the model
        '''
        dataloader = AdversarialDataloader(self.dataset_config, self.model_config['params']['training_params'])
        _, _, test_dataloader = dataloader.get_dataloaders()
        
        model = Module(self.config)
        model.load_checkpoint()
        score = model.score(test_dataloader)
        
        logging.info(score)

class AttackRunner():
    '''
    Attack runner, load model, attack it and evaluate on attacked data
    '''
    def __init__(self, config: Dict) -> None:
        '''
        Init method AttackRunner
        :params:
            config: Dict - configuration file with parameters
        '''
        self.config = config
        self.model_config = config['model']
        self.dataset_config = config['dataset']
        self.attack_config = config['attack']
    
    def run(self) -> None:
        '''
        Method for attack model
        Initialize dataloaders, attack class and evaluate model on attacked data
        '''
        dataloader = AdversarialDataloader(self.dataset_config, self.model_config['params']['training_params'])
        _, _, test_dataloader = dataloader.get_dataloaders()

        model = Module(self.config)
        model.load_checkpoint()
        score = model.score(test_dataloader)
        logging.info(f'Model performance before attack: {score}')

        if self.attack_config['name'] == 'fgsm_untargeted':
            attacker = FGSMAttack(self.attack_config, model._model)
        elif self.attack_config['name'] == 'pgd_untargeted':
            attacker = PGDAttack(self.attack_config, model._model)
        else:
            raise Exception(f"Unsupported attack: {self.attack_config['name']}")
            
        attacker.fit(test_dataloader)