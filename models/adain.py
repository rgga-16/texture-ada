import torch
import torch.nn as nn
from torch.nn import functional as F
from models.feedforward import FeedForwardNetwork
from models.texture_transfer_models import VGG19,Pyramid2D


class AdaIN(nn.Module):
    def __init__(self):
        super(AdaIN,self).__init__()
    
    def forward(self,X,Y):
        X_aligned = self.adain(X,Y)
        return X_aligned
    
    @staticmethod
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

        x_feats = x.view(b_x,c_x,w_x*h_x)
        x_mean = torch.mean(x_feats,dim=2,keepdim=True)
        x_std = torch.std(x_feats,dim=2,keepdim=True)

        y_feats = y.view(b_y,c_y,w_y*h_y)
        y_mean = torch.mean(y_feats,dim=2,keepdim=True)
        y_std = torch.std(y_feats,dim=2,keepdim=True)

        normalized_x = (x_feats - x_mean) / x_std
        output_feats = (normalized_x * y_std) + y_mean

        output = output_feats.view(b_x,c_x,w_x,h_x)
        return output

def adain(x,y):
        b_x,c_x,w_x,h_x = x.shape
        b_y,c_y,w_y,h_y = y.shape
        assert b_x*c_x == b_y*c_y

        x_feats = x.view(b_x,c_x,w_x*h_x)
        x_mean = torch.mean(x_feats,dim=2,keepdim=True)
        x_std = torch.std(x_feats,dim=2,keepdim=True)

        y_feats = y.view(b_y,c_y,w_y*h_y)
        y_mean = torch.mean(y_feats,dim=2,keepdim=True)
        y_std = torch.std(y_feats,dim=2,keepdim=True)

        normalized_x = (x_feats - x_mean) / x_std
        output_feats = (normalized_x * y_std) + y_mean

        output = output_feats.view(b_x,c_x,w_x,h_x)
        return output

def calc_mean_std(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert (len(size) == 4)
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


"""
Original implementation (in Pytorch) of AdaIN Network

Based on the paper titled,
"Arbitrary Style Transfer in Real-time with Adaptive Instance Normalization"

Code borrowed from: https://github.com/naoto0804/pytorch-AdaIN 
"""
class Network_AdaIN(nn.Module):
    def __init__(self):
        super(Network_AdaIN,self).__init__()
        self.encoder = nn.Sequential(*list(VGG19().features.children())[:21])

        # Download VGG19 normalized.
        for param in self.encoder.parameters():
            param.requires_grad = False

        self.decoder = nn.Sequential(
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(512, 256, (3, 3)),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 256, (3, 3)),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 256, (3, 3)),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 256, (3, 3)),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 128, (3, 3)),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(128, 128, (3, 3)),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(128, 64, (3, 3)),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(64, 64, (3, 3)),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(64, 3, (3, 3)),
        )

        
    
    def forward(self,X,style):
        x_feats = self.encoder(X)
        style_feats = self.encoder(style)

        x_feats = adaptive_instance_normalization(x_feats,style_feats)
        output = self.decoder(x_feats)

        return output

class Pyramid2D_Adain(Pyramid2D):
    def __init__(self, ch_in, ch_step, ch_out, n_samples):
        super().__init__(ch_in=ch_in, ch_step=ch_step, ch_out=ch_out, n_samples=n_samples)
    
    def forward(self, z):
        # Assuming image size = 256x256
        #                       z[0]        z[1]        z[2]        z[3]     z[4]       z[5]   
        # z is list of tensors [(3x256x256),(3x128x128),(3x64x64),(3x32x32),(3x16x16),(3x8x8)]

        y = self.cb1_1(z[5]) # z[5]=(3x8x8) => y=(8x8x8)
        y = self.up1(y) # y=(8x8x8) => y=(8x16x16)

        # z[4]=(3x16x16) => (8x16x16)
        # y cat z[4] => y=(16x16x16)
        y = torch.cat((y,self.cb2_1(z[4])),1)

        y = self.cb2_2(y) # y=(16x16x16) => (16x16x16)
        y = self.up2(y) # y=(16x32x32)

        # z[3]=(3x32x32) => (8x32x32)
        # y cat z[3] => (24x32x32)
        y = torch.cat((y,self.cb3_1(z[3])),1)

        y = self.cb3_2(y) # y=(24x32x32) => (24x32x32)
        y = self.up3(y) # y=(24x32x32) => (24x64x64)

        # z[2]=(3x64x64) => (8x64x64)
        # y cat z[2] => (32x64x64)
        y = torch.cat((y,self.cb4_1(z[2])),1)

        y = self.cb4_2(y) # y=(32x64x64) => (32x64x64)
        # Insert adain here??
        y = self.up4(y) # y=(32x128x128)
        # Or insert adain here??

        # z[1]=(3x128x128) => (8x128x128)
        # y cat z[1] => (40x128x128)
        y = torch.cat((y,self.cb5_1(z[1])),1)


        y = self.cb5_2(y) # y=(40x128x128) => (40x128x128)
        y = self.up5(y) # y=(40x128x128) => (40x256x256)

        # z[0]=(3x256x256) => (8x256x256)
        # y cat z[0] => (48x256x256)
        y = torch.cat((y,self.cb6_1(z[0])),1)

        y = self.cb6_2(y) # y=(48x256x256) => (48x256x256)
        y = self.last_conv(y) # y=(48x256x256) => (3x256x256)
        return y


"""
Extended version of Feedforward Style Transfer Network with AdaIN layers added.

Based on the paper titled,
"Perceptual Losses for Real-Time Style Transfer and Super-Resolution" (2016) by Johnson et al.

Code borrowed from: https://github.com/pytorch/examples/blob/master/fast_neural_style/neural_style/transformer_net.py
"""
class FeedForwardNetwork_AdaIN(FeedForwardNetwork):

    def __init__(self,in_channels=3,out_channels=3,n_resblocks = 5) -> None:
        super(FeedForwardNetwork_AdaIN,self).__init__()
        FeedForwardNetwork.__init__(self, in_channels,out_channels,n_resblocks)
        self.adain_layer = AdaIN()
    
    def forward(self,X,style):
        y = self.relu(self.in1(self.conv1(X)))
        y = self.relu(self.in2(self.conv2(y)))
        y = self.relu(self.in3(self.conv3(y)))

        style_y = self.relu(self.conv1(style))
        style_y = self.relu(self.conv2(style_y))
        style_y = self.relu(self.conv3(style_y))

        y = self.res1(y)
        y = self.res2(y)
        y = self.res3(y)
        y = self.res4(y)
        y = self.res5(y)

        y = self.relu(self.in4(self.deconv1(y)))
        y = self.relu(self.in5(self.deconv2(y)))
        y = self.deconv3(y)
        return y

