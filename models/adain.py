import torch
import torch.nn as nn
from torch.nn import functional as F
from models.feedforward import FeedForwardNetwork
from models.texture_transfer_models import VGG19,Pyramid2D


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

