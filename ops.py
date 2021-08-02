import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import numpy as np

from helpers import image_utils
from defaults import DEFAULTS as D

def calc_mean_std(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    
    size = feat.size()
    assert (len(size) == 4 or len(size) == 3 )
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std


def adaptive_instance_normalization(content_feat, style_feat):
    assert (content_feat.size()[:2] == style_feat.size()[:2])
    size = content_feat.size()
    style_mean, style_std = calc_mean_std(style_feat)
    content_mean, content_std = calc_mean_std(content_feat)

    normalized_feat = (content_feat - content_mean.expand(
        size)) / content_std.expand(size)
    return normalized_feat * style_std.expand(size) + style_mean.expand(size)

def adain_pointcloud(content_feat, style_feat):
    assert (content_feat.size()[:2] == style_feat.size()[:2])
    bs,c = content_feat.size()[:2]
    style_mean, style_std = calc_mean_std(style_feat)
    content_mean, content_std = calc_mean_std(content_feat)

    normalized_feat = (content_feat - content_mean.reshape(bs,c,-1)) / content_std.reshape(bs,c,-1)
    return normalized_feat * style_std.reshape(bs,c,-1) + style_mean.reshape(bs,c,-1)

def gram_matrix(tensor):
    b,c,w,h = tensor.shape
    features = tensor.view(b*c,-1)
    gram = torch.mm(features,features.t())
    return gram / (h*w)

def get_mean(tensor):
    b,c,w,h = tensor.shape
    feats = tensor.view(b,c,h*w)
    mean=torch.mean(feats,dim=2,keepdim=True)
    return mean 

def covariance_matrix(tensor):
    b,c,w,h = tensor.shape
    feats = tensor.view(b,c,h*w)
    mean=torch.mean(feats,dim=2,keepdim=True)
    feats=feats-mean
    feats = feats.squeeze()
    covariance = torch.matmul(feats,torch.transpose(feats,-2,-1))
    return covariance / (h*w)

'''
Code borrowed from: https://github.com/VinceMarron/style_transfer/blob/master/why_wasserstein.ipynb
'''
def gaussian_wasserstein_distance(mean1,cov1,mean2,cov2):
    mean1=mean1.detach()
    cov1 = cov1.detach()
    mean2= mean2.detach()
    cov2 = cov2.detach()

    mean_diff_pt = torch.sum((mean1-mean2)**2)
    var_components_pt = cov1.diagonal(offset=0,dim1=-1,dim2=-2).sum(-1)
    var_overlap_pt = torch.sum(torch.sqrt(torch.linalg.eigvals(torch.matmul(cov1,cov2))))
    
    return  torch.sqrt(mean_diff_pt+var_components_pt-2*var_overlap_pt)

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

