import torch
import torch.nn as nn
import numpy as np
import logging
from sklearn.metrics import accuracy_score

from typing import Dict
from torch.utils.data import DataLoader

class FGSMAttack():
    '''
    Method from "EXPLAINING AND HARNESSING ADVERSARIAL EXAMPLES":
    https://arxiv.org/pdf/1412.6572
    '''
    def __init__(self, attack_config: Dict, model: nn.Module) -> None:
        '''
        Init method FGSMAttack
        :params:
            attack_config: Dict - configuration dictionary with params
            model: nn.Module - model for apply attack
        '''
        self._model = model
        self._device = next(model.parameters()).device
        self._criterion = nn.CrossEntropyLoss()

        self._epsilon = attack_config['params']['epsilon']
    
    def attack(self, x: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        '''
        Method for apply attack on data
        '''
        x = x.requires_grad_(True)
        out = self._model(x)
        loss = self._criterion(out, label)
        loss.backward()

        x = x + self._epsilon * torch.sign(x.grad)
        return x.detach()
    
    def fit(self, dataloader: DataLoader) -> None:
        '''
        Method for run attack and compute score on attacked data
        '''
        y_predicted = []
        y_true = []
        dif_data = []
        self._model.eval()
        for batch in dataloader:
            x_attacked = self.attack(batch[0].to(self._device), batch[1].to(self._device))
            out = self._model(x_attacked)
            perturbation = (x_attacked.cpu() - batch[0]).detach()

            dif_data.append(torch.sqrt(torch.mean(perturbation**2)).numpy())
            y_predicted.append(out.argmax(1).detach().cpu().numpy())
            y_true.append(batch[1].numpy())

        # compute score
        y_predicted = np.concatenate(y_predicted)
        y_true = np.concatenate(y_true)
        x_norm = np.mean(dif_data)
        accuracy_attacked = accuracy_score(y_true, y_predicted)

        logging.info(f"Accuracy after attack: {accuracy_attacked}, perturbation norm: {x_norm}")