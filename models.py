import torch
import torch.nn as nn
from torchvision import models, transforms

from defaults import DEFAULTS as D

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

            if layers is not None and name in layers:
 
                extracted_feats[layers[name]]=x
        
        if layers:
            return extracted_feats
        return x


class ConvAutoencoder(nn.Module):

    def __init__(self,filter_size=3,stride=1,padding=0,pmode='reflect'):
        super(ConvAutoencoder,self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(3,6,kernel_size=filter_size,stride=stride),
            nn.InstanceNorm2d(num_features=6),
            nn.ReLU(inplace=True),
            nn.Conv2d(6,16,kernel_size=filter_size,stride=stride),
            nn.InstanceNorm2d(num_features=16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16,32,kernel_size=3,stride=stride),
            nn.InstanceNorm2d(num_features=32),
            nn.ReLU(inplace=True),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32,16,kernel_size=filter_size,stride=stride),
            nn.InstanceNorm2d(16),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(16,6,kernel_size=filter_size,stride=stride),
            nn.InstanceNorm2d(6),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(6,3,kernel_size=3,stride=stride),
            nn.InstanceNorm2d(3),
            nn.ReLU(inplace=True),
        )

    def forward(self,x):
        y=self.encoder(x)
        z=self.decoder(y)

        return z

class TN_FullConvLayer(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(TN_FullConvLayer,self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(ch_in,ch_out,kernel_size=3,padding=1,stride=1,padding_mode='reflect'),
            nn.InstanceNorm2d(num_features=ch_out),
            nn.ReLU(inplace=True),
        )

    def forward(self,x):
        y=self.layer(x)
        return y

class TN_UpsampleLayer(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(TN_UpsampleLayer,self).__init__()
        self.layer = nn.Sequential(
            nn.Upsample(scale_factor=2,mode='bilinear'),
            nn.Conv2d(ch_in,ch_out,kernel_size=3,padding=1),
            nn.InstanceNorm2d(num_features=ch_out),
            nn.ReLU(inplace=True),
        )
    
    def forward(self,x):
        y=self.layer(x)
        return y


class TextureNet(nn.Module):
    def __init__(self):
        super(TextureNet,self).__init__()

        self.convs = nn.Sequential(
            TN_FullConvLayer(3,32),
            TN_FullConvLayer(32,64),
            TN_FullConvLayer(64,128),
            # TN_FullConvLayer(128,128),
            # TN_FullConvLayer(128,128),
        )
        self.upsamplers = nn.Sequential(
            TN_UpsampleLayer(128,64),
            TN_UpsampleLayer(64,32),
            TN_UpsampleLayer(32,3),
        )

    def forward(self,x):
        x=self.convs(x)
        x=self.upsamplers(x)
        return x


