import torch
import torch.nn as nn
import numpy as np
import logging
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

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
        for batch in tqdm(dataloader):
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

        ### illustrate work

        num_examples = 6
        indeces = np.random.choice(np.arange(len(batch[0])), num_examples, replace=False)
        data = batch[0]
        label = batch[1]
        plt.subplots(num_examples, 3, figsize=(5,18))

        for i, idx in enumerate(indeces):
            plt.subplot(num_examples, 3, (i*3)+1)
            plt.title(f"True label: {label[idx]}")
            plt.imshow(data[idx,0], cmap='gray', vmin=0.0, vmax=1.0)
            plt.axis(False)

            plt.subplot(num_examples, 3, (i*3)+2)
            plt.title('Perturbation')
            plt.imshow(perturbation[idx,0].detach().cpu().numpy(), cmap='gray', vmin=0.0, vmax=1.0)
            plt.axis(False)

            plt.subplot(num_examples, 3, (i*3)+3)
            plt.title(f"Prediction: {out.argmax(1).detach().cpu().numpy()[idx]}")
            plt.imshow(x_attacked[idx,0].detach().cpu().numpy(), cmap='gray', vmin=0.0, vmax=1.0)
            plt.axis(False)
        os.makedirs('./artifacts/plots/', exist_ok=True)
        plt.savefig('./artifacts/plots/FGSM attack example.png', bbox_inches="tight")

class PGDAttack():
    '''
    '''
    def __init__(self, attack_config: Dict, model: nn.Module) -> None:
        '''
        Init method PGDAttac
        :params:
            attack_config: Dict - configuration dictionary with params
            model: nn.Module - model for apply attack
        '''
        self._model = model
        self._device = next(model.parameters()).device
        self._criterion = nn.CrossEntropyLoss()

        self._epsilon = attack_config['params']['epsilon']
        self._num_iters = attack_config['params']['num_iters']
    
    def attack(self, x: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        '''
        Method for apply attack on data
        '''
        x_attacked = x.clone()

        for iter in range(self._num_iters):
            x_attacked.requires_grad_(True)
            out = self._model(x_attacked)
            loss = self._criterion(out, label)

            self._model.zero_grad()
            loss.backward()

            with torch.no_grad():
                x_attacked = x_attacked + self._epsilon * torch.sign(x_attacked.grad)
                perturbation = torch.clamp(x_attacked-x, min=-self._epsilon, max=self._epsilon)
                x_attacked = torch.clamp(x+perturbation, min=0, max=1)

        return x_attacked.detach()
    
    def fit(self, dataloader: DataLoader) -> None:
        '''
        Method for run attack and compute score on attacked data
        '''
        y_predicted = []
        y_true = []
        dif_data = []
        self._model.eval()
        for batch in tqdm(dataloader):
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

        ### illustrate work

        num_examples = 6
        indeces = np.random.choice(np.arange(len(batch[0])), num_examples, replace=False)
        data = batch[0]
        label = batch[1]
        plt.subplots(num_examples, 3, figsize=(5,18))

        for i, idx in enumerate(indeces):
            plt.subplot(num_examples, 3, (i*3)+1)
            plt.title(f"True label: {label[idx]}")
            plt.imshow(data[idx,0], cmap='gray', vmin=0.0, vmax=1.0)
            plt.axis(False)

            plt.subplot(num_examples, 3, (i*3)+2)
            plt.title('Perturbation')
            plt.imshow(perturbation[idx,0].detach().cpu().numpy(), cmap='gray', vmin=0.0, vmax=1.0)
            plt.axis(False)

            plt.subplot(num_examples, 3, (i*3)+3)
            plt.title(f"Prediction: {out.argmax(1).detach().cpu().numpy()[idx]}")
            plt.imshow(x_attacked[idx,0].detach().cpu().numpy(), cmap='gray', vmin=0.0, vmax=1.0)
            plt.axis(False)
        os.makedirs('./artifacts/plots/', exist_ok=True)
        plt.savefig('./artifacts/plots/PGD attack example.png', bbox_inches="tight")