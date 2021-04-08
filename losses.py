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


def adain(x,y):
    """
    Aligns mean and std of feature maps x to that of feature maps y.
    Based on the AdaIN formula introduced by Huang & Belongie (2017) in their paper,
    "Arbitrary Style Transfer in Real-time with Adaptive Instance Normalization".
    Link: https://arxiv.org/pdf/1703.06868.pdf

    :param x: (b,c,h_x,w_x), feature maps x
    :param x: (b,c,h_y,w_y), feature maps y
    :return: (b,c,h_x,w_x), feature maps x with mean and variance matching y
    """
    b_x,c_x,w_x,h_x = x.shape
    b_y,c_y,w_y,h_y = y.shape

    assert b_x*c_x == b_y*c_y

    x_feats = x.clone().view(b_x,c_x,w_x*h_x)
    x_mean = torch.mean(x_feats,dim=2,keepdim=True)
    x_std = torch.std(x_feats,dim=2,keepdim=True)

    y_feats = y.clone().view(b_y,c_y,w_y*h_y)
    y_mean = torch.mean(y_feats,dim=2,keepdim=True)
    y_std = torch.std(y_feats,dim=2,keepdim=True)

    normalized_x = (x_feats - x_mean) / x_std
    output_feats = (normalized_x * y_std) + y_mean

    output = output_feats.view(b_x,c_x,w_x,h_x)
    return output
    



def covariance_matrix(tensor):
    b,c,w,h = tensor.shape
    feats = tensor.view(b,c,h*w)
    mean=torch.mean(feats,dim=2,keepdim=True)
    feats=feats-mean
    feats = feats.squeeze()
    covariance = torch.mm(feats,feats.t())
    return covariance / (h*w)

def variance_aware_adaptive_weighting(tensor):

    return


# def weighted_style_rep(tensor):
#     b,c,w,h = tensor.shape
#     feats = tensor.view(b*c,h*w)
#     var = torch.var(feats,dim=1,keepdim=True)
    
#     w_style = (1.0/var.inverse()) * covariance_matrix(tensor)
#     return w_style


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

