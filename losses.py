import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

from helpers import utils
from defaults import DEFAULTS as D

class Normalization(nn.Module):
    def __init__(self,mean,std):
        # .view the mean and std to make them [C x 1 x 1] so that they can
        # directly work with image Tensor of shape [B x C x H x W].
        # B is batch size. C is number of channels. H is height and W is width.
        super(Normalization,self).__init__()
        device = D.DEVICE()
        self.mean = torch.tensor(mean).view(-1,1,1).to(device)
        self.std = torch.tensor(std).view(-1,1,1).to(device)

    def forward(self, img):
        return (img-self.mean)/self.std

def gram_matrix(tensor):
    b,c,w,h = tensor.shape
    features = tensor.view(b*c,-1)
    gram = torch.mm(features,features.t())
    return gram / (h*w)

def covariance_matrix(tensor):
    b,c,w,h = tensor.shape
    feats = tensor.view(b*c,h*w)
    mean=torch.mean(feats,dim=1,keepdim=True)
    thing=feats-mean
    covariance = torch.mm(thing,thing.t())
    return covariance


def sliced_wasserstein_loss():

    pass

class ContentLoss(nn.Module): 

    def __init__(self,target):
        super(ContentLoss, self).__init__()
        # we 'detach' the target content from the tree used
        # to dynamically compute the gradient: this is a stated value,
        # not a variable. Otherwise the forward method of the criterion
        # will throw an error.
        self.target = target.detach()

    def forward(self,input):
        self.loss= F.mse_loss(input,self.target)
        
        return input 

class StyleLoss(nn.Module):

    def __init__(self,target_feature):
        super(StyleLoss,self).__init__()

        self.target = gram_matrix(target_feature).detach()

    def forward(self, input):
        g = gram_matrix(input)
        self.loss = F.mse_loss(g,self.target)

        return input

