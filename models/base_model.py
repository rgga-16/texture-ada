import torch
from torch import nn
import numpy as np
from torch.nn import functional as F
import torchvision
from defaults import DEFAULTS as D

from abc import ABC, abstractmethod

class BaseModel(ABC):
    def __init__(self,net,optimizer,criterion_loss):
        self.device = D.DEVICE()
        self.net = net.to(self.device)
        self.optimizer = optimizer
        self.criterion_loss = criterion_loss.to(self.device)
    
    def get_state_dict(self):
        return {
            'model_state_dict':self.net.state_dict(),
            'optimizer_state_dict':self.optimizer.state_dict(),
        }

    def load_state_dict(self,state_dict):
        self.net.load_state_dict(state_dict['model_state_dict'])
        self.optimizer.load_state_dict(state_dict['optimizer_state_dict'])
    
    @abstractmethod
    def set_input(self):
        pass

    @abstractmethod
    def forward(self):
        pass

    @abstractmethod
    def get_losses(self):
        pass

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def eval(self):
        pass

    @abstractmethod
    def optimize_parameters(self):
        pass
