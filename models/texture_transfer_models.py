import torch
import torch.nn as nn
from torch.nn import functional as F
from torchvision import models, transforms

from defaults import DEFAULTS as D
import copy


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

class VGG19(nn.Module):

    def __init__(self,vgg_path='models/weights/vgg19-dcbb9e9d.pth',device=D.DEVICE()):
        super(VGG19,self).__init__()

        _ = models.vgg19(pretrained=True).eval().to(device)
        # _.load_state_dict(torch.load(vgg_path), strict=False)
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



class Conv_block2D(nn.Module):
    def __init__(self, n_ch_in, n_ch_out, m=0.1):
        super(Conv_block2D, self).__init__()

        self.conv1 = nn.Conv2d(n_ch_in, n_ch_out, 3, padding=0, bias=True)
        # self.bn1 = nn.InstanceNorm2d(num_features=n_ch_out,momentum=m)
        self.conv2 = nn.Conv2d(n_ch_out, n_ch_out, 3, padding=0, bias=True)
        # self.bn2 = nn.InstanceNorm2d(num_features=n_ch_out,momentum=m)
        self.conv3 = nn.Conv2d(n_ch_out, n_ch_out, 1, padding=0, bias=True)
        # self.bn3 = nn.InstanceNorm2d(num_features=n_ch_out,momentum=m)

    def forward(self, x):
        # Pad x with its top and bottom pixel layers
        x = torch.cat((x[:,:,-1,:].unsqueeze(2),x,x[:,:,0,:].unsqueeze(2)),2)
        # Pad x with its left and right pixel layers
        x = torch.cat((x[:,:,:,-1].unsqueeze(3),x,x[:,:,:,0].unsqueeze(3)),3)
        # x = F.leaky_relu(self.bn1(self.conv1(x)))
        x = F.leaky_relu(self.conv1(x))
         # Pad x with its top and bottom pixel layers
        x = torch.cat((x[:,:,-1,:].unsqueeze(2),x,x[:,:,0,:].unsqueeze(2)),2)
        # Pad x with its left and right pixel layers
        x = torch.cat((x[:,:,:,-1].unsqueeze(3),x,x[:,:,:,0].unsqueeze(3)),3)
        # x = F.leaky_relu(self.bn2(self.conv2(x)))
        # x = F.leaky_relu(self.bn3(self.conv3(x)))
        x = F.leaky_relu(self.conv2(x))
        x = F.leaky_relu(self.conv3(x))
        return x

#Up-sampling + instance normalization block
class Up_In2D(nn.Module):
    def __init__(self, n_ch):
        super(Up_In2D, self).__init__()

        self.up = nn.Upsample(scale_factor=2, mode='nearest')
        # self.inst_norm = nn.InstanceNorm2d(num_features=n_ch)

    def forward(self, x):
        # x = self.inst_norm(self.up(x))
        x = self.up(x)
        return x

class Pyramid2D(nn.Module):
    def __init__(self, ch_in=3, ch_step=8, ch_out=3, n_samples=6):
        super(Pyramid2D, self).__init__()

        self.conv_blocks = []
        self.conv_entries = []

        for i in range(n_samples):
            running_step = ch_step * (i+1)
            self.conv_blocks.append(
                nn.Sequential(
                    Conv_block2D(ch_in,running_step) if i==0 
                    else Conv_block2D(running_step,running_step),
                    Up_In2D(running_step),
                )
            )
            if i > 0:
                self.conv_entries.append(
                    Conv_block2D(ch_in,ch_step)
                )

        

        self.cb1_1 = Conv_block2D(ch_in,ch_step) # ch_step=8
        self.up1 = Up_In2D(ch_step)

        self.cb2_1 = Conv_block2D(ch_in,ch_step)
        self.cb2_2 = Conv_block2D(2*ch_step,2*ch_step) # ch_step=16
        self.up2 = Up_In2D(2*ch_step)

        self.cb3_1 = Conv_block2D(ch_in,ch_step)
        self.cb3_2 = Conv_block2D(3*ch_step,3*ch_step) # ch_step=24
        self.up3 = Up_In2D(3*ch_step)

        self.cb4_1 = Conv_block2D(ch_in,ch_step)
        self.cb4_2 = Conv_block2D(4*ch_step,4*ch_step) # ch_step=32
        self.up4 = Up_In2D(4*ch_step)

        self.cb5_1 = Conv_block2D(ch_in,ch_step)
        self.cb5_2 = Conv_block2D(5*ch_step,5*ch_step) # ch_step=40
        self.up5 = Up_In2D(5*ch_step)

        self.cb6_1 = Conv_block2D(ch_in,ch_step)
        self.cb6_2 = Conv_block2D(6*ch_step,6*ch_step) # ch_step=48
        self.last_conv = nn.Conv2d(6*ch_step, ch_out, 1, padding=0, bias=True)

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
        y = self.up4(y) # y=(32x128x128)

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

        # def forward(self,z):
        #     y = None
        #     # Order of z must start with the smallest samples
        #     for i in range(len(z)):
        #         y = self.conv_blocks[i](z[i])
        #         if i == len(z)-1:
        #             y = torch.cat((y,self.conv_entries[i](z[i+1])))
        #     return 

class Pyramid2D_custom(nn.Module):
    def __init__(self, ch_in=3, ch_step=8, ch_out=3, n_samples=6):
        super(Pyramid2D_custom, self).__init__()
        self.n_samples = n_samples
        conv_blocks = []
        conv_entries = []

        for i in range(n_samples):
            running_step = ch_step * (i+1)
            list_ = [Conv_block2D(ch_in,running_step) if i==0 else Conv_block2D(running_step,running_step)]
            if i != n_samples-1:
                list_.append(Up_In2D(running_step))
            conv_block = nn.Sequential(*list_)
            conv_blocks.append(conv_block)
            if i > 0:
                conv_entries.append(
                    Conv_block2D(ch_in,ch_step)
                )
        self.conv_blocks = nn.ModuleList(conv_blocks)
        self.conv_entries = nn.ModuleList(conv_entries)
        self.final_conv = nn.Conv2d(n_samples*ch_step, ch_out, 1, padding=0, bias=True)

    def forward(self, z):
        assert len(z) == self.n_samples
        inputs = z
        inputs.reverse()
        y = inputs[0]
        for i in range(self.n_samples):
            y = self.conv_blocks[i](y)
            if i != (self.n_samples-1):
                y = torch.cat((y,self.conv_entries[i](inputs[i+1])),1)
        y = self.final_conv(y)
        return y 

class Pyramid2D_adain(nn.Module):
    def __init__(self, ch_in=3, ch_step=8, ch_out=3, n_samples=6):
        super(Pyramid2D_adain, self).__init__()
        self.n_samples = n_samples
        conv_blocks = []
        conv_entries = []
        upsamples = []

        for i in range(n_samples):
            running_step = ch_step * (i+1)
            list_ = [Conv_block2D(ch_in,running_step) if i==0 else Conv_block2D(running_step,running_step)]
            if i != n_samples-1:
                upsamples.append(Up_In2D(running_step))
            conv_block = nn.Sequential(*list_)
            conv_blocks.append(conv_block)
            if i > 0:
                conv_entries.append(
                    Conv_block2D(ch_in,ch_step)
                )

        self.encoder = nn.Sequential(*list(VGG19().features.children())[:18])

        for param in self.encoder.parameters():
            param.requires_grad = False

        self.conv_blocks = nn.ModuleList(conv_blocks)
        self.upsamples = nn.ModuleList(upsamples)
        self.conv_entries = nn.ModuleList(conv_entries)

        self.final_conv = nn.Conv2d(n_samples*ch_step, ch_out, 1, padding=0, bias=True)

    def forward(self, z, style):
        assert len(z) == self.n_samples
        inputs = z
        inputs.reverse()
        y = inputs[0]

        style_feats = self.encoder(style)

        # adain.adaptive_instance_normalization()
        for i in range(self.n_samples):
            y = self.conv_blocks[i](y)

            if i==3:
                # Perform adain operation here
                y = adaptive_instance_normalization(y, style_feats)

            if i != (self.n_samples-1):
                y = self.upsamples[i](y)
                y = torch.cat((y,self.conv_entries[i](inputs[i+1])),1)
        y = self.final_conv(y)
        return y 
