import torch
import torch.nn as nn
from torch.nn import functional as F
from models.feedforward import FeedForwardNetwork


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


"""
Original implementation (in Pytorch) of AdaIN Network

Based on the paper titled,
"Arbitrary Style Transfer in Real-time with Adaptive Instance Normalization"

Code borrowed from: https://github.com/naoto0804/pytorch-AdaIN 
"""
class Network_AdaIN(nn.Module):
    def __init__(self):
        super(Network_AdaIN,self).__init__()


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
        y = self.relu(self.conv1(X))
        y = self.relu(self.conv2(y))
        y = self.relu(self.conv3(y))

        style_y = self.relu(self.conv1(style))
        style_y = self.relu(self.conv2(style_y))
        style_y = self.relu(self.conv3(style_y))

        y = self.adain_layer(y,style_y)

        y = self.res1(y)
        y = self.res2(y)
        y = self.res3(y)
        y = self.res4(y)
        y = self.res5(y)

        y = self.relu(self.in4(self.deconv1(y)))
        y = self.relu(self.in5(self.deconv2(y)))
        y = self.deconv3(y)
        return y

