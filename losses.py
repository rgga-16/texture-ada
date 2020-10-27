import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

import utils

class Normalization(nn.Module):
    def __init__(self,mean,std):
        # .view the mean and std to make them [C x 1 x 1] so that they can
        # directly work with image Tensor of shape [B x C x H x W].
        # B is batch size. C is number of channels. H is height and W is width.
        super(Normalization,self).__init__()
        device = utils.setup_device()
        self.mean = torch.tensor(mean).view(-1,1,1).to(device)
        self.std = torch.tensor(std).view(-1,1,1).to(device)

    def forward(self, img):
        return (img-self.mean)/self.std

def gram_matrix(tensor):
    # Get no. of feature maps, feature map width and feature map height
    _,n,w,h = tensor.shape

    features = tensor.view(n,w*h)

    gram = torch.mm(features,features.t())

    return gram / (h*w)

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


content_layers_default = ['conv_4']
style_layers_default = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']
normalization_mean_default = [0.485,0.456,0.406]
normalization_std_default = [0.229,0.224,0.225]

