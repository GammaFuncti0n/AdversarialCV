import torch
import torch.nn as nn
import numpy as np
import os
import yaml
import logging
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from typing import Tuple

from adversarialcv.model import SimpleCNN

class Module():
    def __init__(self, config)->None:
        '''
        Init method Module
        '''
        # set configs
        self.config = config
        self.model_config = config['model']
        self.dataset_config = config['dataset']

        # set model
        self._model_name = self.model_config['name']

        # parameters
        self._model_params = self.model_config['params']['model_params']
        self._train_parameters = self.model_config['params']['training_params']
        self._device = config['env']['device']

        # init model and utils for train
        self._model = self.__init_model()
        self._criterion = self.__init_criterion()
        self._optimizer = self.__init_optimizer()
        self._scheduler = self.__init_scheduler()
    
    def __init_model(self):
        '''
        Initializing model
        '''
        if self._model_name == 'simple_cnn':
            return SimpleCNN(**self._model_params).to(self._device)
        else:
            raise Exception(f"Unsupported model: {self._model_name}")
    
    def __init_criterion(self):
        '''
        Initializing criterion
        '''
        if self._train_parameters['criterion'] == 'cross_entropy_loss':
            return torch.nn.CrossEntropyLoss()
        else:
            raise Exception(f"Unsupported criterion: {self._train_parameters['criterion']}")
    
    def __init_optimizer(self):
        '''
        Initializing optimizer
        '''
        if self._train_parameters['optimizer']['name'] == 'adamw':
            lr = float(self._train_parameters['optimizer']['params']['lr'])
            weight_decay = float(self._train_parameters['optimizer']['params']['weight_decay'])
            return torch.optim.AdamW(self._model.parameters(), lr=lr, weight_decay=weight_decay)
        else:
            raise Exception(f"Unsupported optimizer: {self._train_parameters['optimizer']['name']}")
    
    def __init_scheduler(self):
        '''
        Initializing learning rate scheduler
        '''
        if self._train_parameters['scheduler']['name'] == 'exponential_lr':
            return torch.optim.lr_scheduler.ExponentialLR(self._optimizer, **self._train_parameters['scheduler']['params'])
        else:
            raise Exception(f"Unsupported scheduler: {self._train_parameters['scheduler']['name']}")

    def _compute_score(self, y_pred, y_true):
        return accuracy_score(y_true=y_true, y_pred=y_pred)

    def fit(self, train_dataloader, val_dataloader=None)->None:
        '''
        Fit model
        '''
        best_loss = np.inf
        self._num_epochs = self._train_parameters['num_epochs']
        for self._epoch in range(self._num_epochs):
            # Train epoch
            train_loss, train_score = self._train_epoch(train_dataloader)
            train_message = f"train loss = {train_loss:.4f}, train_score = {train_score:.3f}"
            if(self._epoch%self._train_parameters['scheduler']['step']):
                self._scheduler.step()

            # Val epoch
            if val_dataloader is not None:
                val_loss, val_score = self._val_epoch(val_dataloader)
                val_message = f", val loss = {val_loss:.4f}, val_score = {val_score:.3f}"
            else:
                val_loss = np.inf
                val_message = ''
            
            # Logging, printing
            message = f"{self._epoch+1}/{self._num_epochs}: " + train_message + val_message
            logging.info(message)

            # Save best model
            if(val_loss < best_loss):
                best_loss = val_loss
                checkpoint_directory = os.path.join(self.config['paths']['artifacts'], 'checkpoints')
                os.makedirs(checkpoint_directory, exist_ok=True)
                torch.save(
                    {
                        'epoch': self._epoch,
                        'model_state_dict': self._model.state_dict(),
                        'loss': best_loss,
                        'model_config': self.model_config,
                    }, 
                    os.path.join(checkpoint_directory, f"{self.model_config['save_name']}.pt")
                    )
                logging.info('Save checkpoint.')
    
    def _train_epoch(self, train_dataloader) -> Tuple[float, float]:
        train_loss_list = []
        y_predicted = []
        y_true = []
        self._model.train()

        pbar = tqdm(total=len(train_dataloader), desc=f'Epoch {self._epoch+1}/{self._num_epochs}', postfix={'loss': '?'}) 
        for batch in train_dataloader:
            self._optimizer.zero_grad()
            out = self._model(batch[0].to(self._device))

            # lists for compute score
            y_predicted.append(out.argmax(1).detach().cpu().numpy())
            y_true.append(batch[1].numpy())

            loss = self._criterion(out, batch[1].to(self._device))
            loss.backward()

            #torch.nn.utils.clip_grad_norm_(self._model.parameters(), self._clip_grad)

            train_loss_list.append(loss.item())
            self._optimizer.step()

            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
            pbar.update(1)
        pbar.close()

        y_predicted = np.concatenate(y_predicted)
        y_true = np.concatenate(y_true)

        return np.mean(train_loss_list), self._compute_score(y_predicted, y_true)

    def _val_epoch(self, val_dataloader) -> Tuple[float, float]:
        val_loss_list = []
        y_predicted = []
        y_true = []
        self._model.eval()
        with torch.no_grad():
            for batch in val_dataloader:
                self._optimizer.zero_grad()
                out = self._model(batch[0].to(self._device))

                # lists for compute score
                y_predicted.append(out.argmax(1).detach().cpu().numpy())
                y_true.append(batch[1].numpy())

                loss = self._criterion(out, batch[1].to(self._device))
                val_loss_list.append(loss.item())
        
        y_predicted = np.concatenate(y_predicted)
        y_true = np.concatenate(y_true)

        return np.mean(val_loss_list), self._compute_score(y_predicted, y_true)