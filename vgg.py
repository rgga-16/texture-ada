import torch
import torch.nn as nn
from torchvision import models, transforms

from defaults import DEFAULTS as D

class VGG16(nn.Module):
    def __init__(self, vgg_path='models/vgg16-00b39a1b.pth'):
        super(VGG16, self).__init__()

        # Load VGG Skeleton, Pretrained Weights
        vgg16_features = models.vgg16(pretrained=False)
        vgg16_features.load_state_dict(torch.load(vgg_path), strict=False)
        self.features = vgg16_features.features

        # Turn-off Gradient History
        for param in self.features.parameters():
            param.requires_grad = False
    
    def forward(self, x):
        # layers = {'3': 'relu1_2', '8': 'relu2_2', '15': 'relu3_3', '22': 'relu4_3'}
        layers = {'1':'relu1_1', '15':'relu3_3', '25':'relu5_1','30':'maxpool'}
        features = {}
        for name, layer in self.features._modules.items():
            x = layer(x)
            if name in layers:
                features[layers[name]] = x
                if (name=='30'):
                    break

        return features

class VGG19(nn.Module):
    def __init__(self,vgg_path='models/vgg19-dcbb9e9d.pth',device=D.DEVICE()):
        super(VGG19,self).__init__()

        _ = models.vgg19(pretrained=False).eval().to(device)
        _.load_state_dict(torch.load(vgg_path), strict=False)
        self.features = _.features

        for param in self.features.parameters():
            param.requires_grad = False
    
    def forward(self,x,layers:dict=None):
        extracted_feats = {}
        for name, layer in self.features._modules.items():
            x = layer(x)

            if name in layers:
                extracted_feats[layers[name]]=x
        
        if layers:
            return x, extracted_feats
        return x


