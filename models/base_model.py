import torch
from torch import nn
import numpy as np
from torch.nn import functional as F
import torchvision
from defaults import DEFAULTS as D

from abc import ABC, abstractmethod

class BaseModel(ABC):
    def __init__(self):
        self.device = D.DEVICE()

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