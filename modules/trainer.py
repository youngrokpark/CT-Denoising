import os
import re
import shutil
import random
import pickle
from os.path import isfile, join
from tqdm.auto import tqdm
from itertools import chain
import numpy as np
import torch
import itertools
from torch.utils.tensorboard import SummaryWriter
from modules.models import ResnetGenerator, Discriminator
from modules.dataset import make_dataloader
from modules.utils import Mean, init_weights, set_requires_grad

import torch.nn as nn

from os import makedirs
from os.path import isdir


class Trainer:
    def __init__(self, train_config, model_config, device):
        self.train_config = train_config
        self.model_config = model_config
        self.device = device
        self._init_models()
        self._init_optimizers()
        self._init_loss()
        
        self._init_path()
        
    def _init_path(self):   
        # check the path existence
        self.path_result = join(self.train_config.path_checkpoint, self.model_config.model_name)    
        if not isdir(join(self.train_config.path_checkpoint)):
            makedirs(join(self.train_config.path_checkpoint))
        if not isdir(join(self.path_result)):
            makedirs(join(self.path_result))
            
        
    def _init_models(self):
        # initalize the models
        c = self.model_config
        
        if self.train_config.supervised:
            self.G_Q2F_sup = ResnetGenerator(
                *c.G_Q2F_sup.args, **c.G_Q2F_sup.kwargs
            ).to(self.device)
            
        else:                
            c = self.model_config
            self.G_F2Q = ResnetGenerator(
                *c.G_F2Q.args, **c.G_F2Q.kwargs
            ).to(self.device)
            self.G_Q2F = ResnetGenerator(
                *c.G_Q2F.args, **c.G_Q2F.kwargs
            ).to(self.device)        
            self.D_F = Discriminator(
                *c.D_F.args, **c.D_F.kwargs
            ).to(self.device)
            self.D_Q = Discriminator(
                *c.D_Q.args, **c.D_Q.kwargs
            ).to(self.device)
    
                
    def _init_optimizers(self):
        # initialize the optimizers
        if self.train_config.supervised:
            # optimizer for supervised learning
            self.G_sup_optim = torch.optim.Adam(
                self.G_Q2F_sup.parameters(), 
                lr=self.train_config.lr, 
                betas=(self.train_config.beta1, self.train_config.beta2)
                )
        else:
            self.G_optim = torch.optim.Adam(
                itertools.chain(self.G_F2Q.parameters(), self.G_Q2F.parameters()), 
                lr=self.train_config.lr, 
                betas=(self.train_config.beta1, self.train_config.beta2)
                )
            self.D_optim = torch.optim.Adam(
                itertools.chain(self.D_F.parameters(), self.D_Q.parameters()),
                lr=self.train_config.lr, 
                betas=(self.train_config.beta1, self.train_config.beta2)
                )
        
        
    def _init_loss(self):
        # initialize the loss functions
        self.adv_loss = nn.MSELoss()
        self.cycle_loss = nn.L1Loss()
        self.iden_loss = nn.L1Loss()
        
        self.loss_name = [
            'G_adv_loss_F',
             'G_adv_loss_Q',
             'G_cycle_loss_F',
             'G_cycle_loss_Q',
             'G_iden_loss_F',
             'G_iden_loss_Q',
             'D_adv_loss_F',
             'D_adv_loss_Q',
             ]
        
    
    def run(self, logger: SummaryWriter=None):
        # run the training
        self._collect_data(self.train_config.path_data)
        if self.train_config.supervised:
            self.load_models_supervised()
        else:
            self.load_models()
            
        pbar = tqdm(range(self.trained_epoch, self.train_config.num_epoch))
        for epoch in pbar:
            if self.train_config.supervised:
                losses = self.train_supervised(epoch)
            else:
                losses = self.train(epoch)
                self.save_models(epoch)
            
            if logger is not None:
                for k, v in losses.items():
                    logger.add_scalar(f'{k}', v, epoch)
                
            pbar.set_description(f'Epoch {epoch}')
    
        
    def train(self, epoch):
        # Implementation for unsupervised training loop
        losses = {name: Mean() for name in self.loss_name}

        for x_F, x_Q, _ in tqdm(self.train_dataloader, desc='Step'):
            x_F = x_F.to(self.device)
            x_Q = x_Q.to(self.device)

            # Set 'requires_grad' of the discriminators as 'False'
            set_requires_grad([self.D_F, self.D_Q], False)

            x_FQ = self.G_F2Q(x_F)
            x_QF = self.G_Q2F(x_Q)
            x_QFQ = self.G_F2Q(x_QF)
            x_FQF = self.G_Q2F(x_FQ)
            x_FF = self.G_Q2F(x_F)
            x_QQ = self.G_F2Q(x_Q)

            pred_D_F_QF = self.D_F(x_QF)
            pred_D_Q_FQ = self.D_Q(x_FQ)

            G_adv_loss_F = self.adv_loss(pred_D_F_QF, torch.ones_like(pred_D_F_QF, device=self.device))
            G_adv_loss_Q = self.adv_loss(pred_D_Q_FQ, torch.ones_like(pred_D_Q_FQ, device=self.device))
            G_cycle_loss_F = self.cycle_loss(x_FQF, x_F)
            G_cycle_loss_Q = self.cycle_loss(x_QFQ, x_Q)
            G_iden_loss_F = self.iden_loss(x_FF, x_F)
            G_iden_loss_Q = self.iden_loss(x_QQ, x_Q)
            G_adv_loss = G_adv_loss_F + G_adv_loss_Q
            G_cycle_loss = G_cycle_loss_F + G_cycle_loss_Q
            G_iden_loss = G_iden_loss_F + G_iden_loss_Q
            G_total_loss = G_adv_loss + self.train_config.lambda_cycle * (G_cycle_loss) + self.train_config.lambda_iden * (G_iden_loss)

            self.G_optim.zero_grad()
            G_total_loss.backward()
            self.G_optim.step()

            # Set 'requires_grad' of the discriminators as 'True'
            set_requires_grad([self.D_F, self.D_Q], True)

            pred_D_F_F = self.D_F(x_F)
            pred_D_Q_Q = self.D_Q(x_Q)
            target_real = torch.ones_like(pred_D_F_F, device=self.device)
            target_fake = torch.zeros_like(pred_D_F_F, device=self.device)

            # You have to detach the outputs of the generators in below codes
            D_adv_loss_F = self.adv_loss(pred_D_F_F, target_real) + self.adv_loss(self.D_F(x_QF.detach()), target_fake)
            D_adv_loss_Q = self.adv_loss(pred_D_Q_Q, target_real) + self.adv_loss(self.D_Q(x_FQ.detach()), target_fake)
            D_total_loss_F = D_adv_loss_F / 2.0
            D_total_loss_Q = D_adv_loss_Q / 2.0

            self.D_optim.zero_grad()
            D_total_loss_F.backward()
            D_total_loss_Q.backward()
            self.D_optim.step()

            # Calculate the average loss during one epoch
            losses['G_adv_loss_F'](G_adv_loss_F.detach())
            losses['G_adv_loss_Q'](G_adv_loss_Q.detach())
            losses['G_cycle_loss_F'](G_cycle_loss_F.detach())
            losses['G_cycle_loss_Q'](G_cycle_loss_Q.detach())
            losses['G_iden_loss_F'](G_iden_loss_F.detach())
            losses['G_iden_loss_Q'](G_iden_loss_Q.detach())
            losses['D_adv_loss_F'](D_adv_loss_F.detach())
            losses['D_adv_loss_Q'](D_adv_loss_Q.detach())
            
        for name in self.loss_name:
            self.losses_list[name].append(losses[name].result())
            torch.save(self.losses_list[name], join(self.path_result, name + '.npy'))
            
        # Plot input/output images every 10 epochs
        if (epoch + 1) % 10 == 0:
            print(f'Epoch: {epoch}, G_adv_loss_F: {losses["G_adv_loss_F"]}, G_adv_loss_Q: {losses["G_adv_loss_Q"]}')

        return {k: v.result() for k, v in losses.items()}
    
    def load_models_supervised(self):
        # Load the trained model for supervised learning
        path_checkpoint = self.train_config.path_checkpoint
        path_result = self.path_result
        model_name = self.model_config.model_name
        
         # Load the last checkpoint if it exists
        if isfile(join(path_checkpoint, f"{model_name}.pth")):
            checkpoint = torch.load(join(path_checkpoint, f"{model_name}.pth"))
            self.G_Q2F_sup.load_state_dict(checkpoint['G_Q2F_sup_state_dict'])
            self.G_sup_optim.load_state_dict(checkpoint['G_sup_optim_state_dict'])
            self.trained_epoch = checkpoint['epoch']
            self.losses_list = {'G_sup_loss': torch.load(join(path_result, 'G_sup_loss' + '.npy'))}
            print('Start from save model - ' + str(self.trained_epoch))
        # If the checkpoint does not exist, start the training with random initialized model
        else:
            init_weights(self.G_Q2F_sup)
            self.trained_epoch = 0
            self.losses_list = {'G_sup_loss': list()}
            print('Start from random initialized model')
    
    def train_supervised(self, epoch):
        # Implementation for supervised training loop
        losses = {'G_sup_loss': Mean()}
        
        path_checkpoint = self.train_config.path_checkpoint
        path_result = self.path_result
        model_name = self.model_config.model_name

        for x_F, x_Q, _ in tqdm(self.train_dataloader, desc='Step'):
            x_F = x_F.to(self.device)
            x_Q = x_Q.to(self.device)

            x_QF = self.G_Q2F_sup(x_Q)
            G_sup_loss = self.iden_loss(x_QF, x_F)

            self.G_sup_optim.zero_grad()
            G_sup_loss.backward()
            self.G_sup_optim.step()

            losses['G_sup_loss'](G_sup_loss.detach())
            
        self.losses_list['G_sup_loss'].append(losses['G_sup_loss'].result())

        # Save the trained model and list of losses
        torch.save({'epoch': epoch + 1,
                    'G_Q2F_sup_state_dict': self.G_Q2F_sup.state_dict(),
                    'G_sup_optim_state_dict': self.G_sup_optim.state_dict()},
                   join(path_checkpoint, f"{model_name}.pth"))
        torch.save(self.losses_list['G_sup_loss'], join(path_result, 'G_sup_loss' + '.npy'))

        # Plot input/output images every 10 epochs
        if (epoch + 1) % 10 == 0:
            print(f'Epoch: {epoch}, G_sup_loss: {losses["G_sup_loss"].result()}')
            
        return {k: v.result() for k, v in losses.items()}
        
    def _collect_data(self, data_path):
        # Load the dataset
        self.train_dataloader, self.test_dataloader = make_dataloader(data_path, self.train_config.batch_size)
        
    def save_models(self, epoch):
        # Save the trained models
        path_checkpoint = self.train_config.path_checkpoint
        model_name = self.model_config.model_name

        torch.save({'epoch': epoch + 1,
                    'G_F2Q_state_dict': self.G_F2Q.state_dict(),
                    'G_Q2F_state_dict': self.G_Q2F.state_dict(),
                    'D_F_state_dict': self.D_F.state_dict(),
                    'D_Q_state_dict': self.D_Q.state_dict(),
                    'G_optim_state_dict': self.G_optim.state_dict(),
                    'D_optim_state_dict': self.D_optim.state_dict()},
                   join(path_checkpoint, model_name + '.pth'))
    
    def load_models(self):
        # Load the trained models
        path_checkpoint = self.train_config.path_checkpoint
        model_name = self.model_config.model_name

        if isfile(join(path_checkpoint, model_name + '.pth')):
            checkpoint = torch.load(join(path_checkpoint, model_name + '.pth'))
            self.G_F2Q.load_state_dict(checkpoint['G_F2Q_state_dict'])
            self.G_Q2F.load_state_dict(checkpoint['G_Q2F_state_dict'])
            self.D_F.load_state_dict(checkpoint['D_F_state_dict'])
            self.D_Q.load_state_dict(checkpoint['D_Q_state_dict'])
            self.G_optim.load_state_dict(checkpoint['G_optim_state_dict'])
            self.D_optim.load_state_dict(checkpoint['D_optim_state_dict'])
            self.trained_epoch = checkpoint['epoch']
            self.losses_list = {name: torch.load(join(self.path_result, name + '.npy')) for name in self.loss_name}
            print('Start from save model - ' + str(self.trained_epoch))
        else:
            init_weights(self.G_F2Q)
            init_weights(self.G_Q2F)
            init_weights(self.D_F)
            init_weights(self.D_Q)
            self.trained_epoch = 0
            self.losses_list = {name: list() for name in self.loss_name}
            print('Start from random initialized model')