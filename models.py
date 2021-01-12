import torch
import torch.nn as nn
from torch.nn import functional as F
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
            nn.Conv2d(ch_in,ch_out,kernel_size=3,padding=0,stride=1),
            nn.InstanceNorm2d(num_features=ch_out),
            nn.ReLU(inplace=True),
        )

        self.conv=nn.Conv2d(ch_in,ch_out,kernel_size=3,padding=0,stride=1)
        self.inst_norm=nn.InstanceNorm2d(num_features=ch_out)
        self.relu=nn.ReLU(inplace=True)

    def forward(self,x):
        # y=self.layer(x)
        x = torch.cat( (x[:,:,-1,:].unsqueeze(2),x,x[:,:,0,:].unsqueeze(2)),dim=2 )
        x = torch.cat( (x[:,:,:,-1].unsqueeze(3),x,x[:,:,:,0].unsqueeze(3)), dim=3 )
        y=self.conv(x[1])
        y=self.inst_norm(y)
        y=self.relu(y)
        return y

class TN_UpsampleLayer(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(TN_UpsampleLayer,self).__init__()
        self.layer = nn.Sequential(
            nn.Upsample(scale_factor=2,mode='nearest'),
            nn.Conv2d(ch_in,ch_out,kernel_size=1,padding=0),
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
            TN_FullConvLayer(128,128),
            TN_FullConvLayer(128,128),
            TN_FullConvLayer(128,128),
            TN_FullConvLayer(128,128),
            TN_FullConvLayer(128,128),
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


class Conv_block2D(nn.Module):
    def __init__(self, n_ch_in, n_ch_out, m=0.1):
        super(Conv_block2D, self).__init__()

        self.conv1 = nn.Conv2d(n_ch_in, n_ch_out, 3, padding=0, bias=True)
        self.bn1 = nn.InstanceNorm2d(num_features=n_ch_out,momentum=m)
        self.conv2 = nn.Conv2d(n_ch_out, n_ch_out, 3, padding=0, bias=True)
        self.bn2 = nn.InstanceNorm2d(num_features=n_ch_out,momentum=m)
        self.conv3 = nn.Conv2d(n_ch_out, n_ch_out, 1, padding=0, bias=True)
        self.bn3 = nn.InstanceNorm2d(num_features=n_ch_out,momentum=m)

    def forward(self, x):
        # Pad x with its top and bottom pixel layers
        x = torch.cat((x[:,:,-1,:].unsqueeze(2),x,x[:,:,0,:].unsqueeze(2)),2)
        # Pad x with its left and right pixel layers
        x = torch.cat((x[:,:,:,-1].unsqueeze(3),x,x[:,:,:,0].unsqueeze(3)),3)
        x = F.leaky_relu(self.bn1(self.conv1(x)))
         # Pad x with its top and bottom pixel layers
        x = torch.cat((x[:,:,-1,:].unsqueeze(2),x,x[:,:,0,:].unsqueeze(2)),2)
        # Pad x with its left and right pixel layers
        x = torch.cat((x[:,:,:,-1].unsqueeze(3),x,x[:,:,:,0].unsqueeze(3)),3)
        x = F.leaky_relu(self.bn2(self.conv2(x)))
        x = F.leaky_relu(self.bn3(self.conv3(x)))
        return x

#Up-sampling + instance normalization block
class Up_In2D(nn.Module):
    def __init__(self, n_ch):
        super(Up_In2D, self).__init__()

        self.up = nn.Upsample(scale_factor=2, mode='nearest')
        self.inst_norm = nn.InstanceNorm2d(num_features=n_ch)

    def forward(self, x):
        x = self.inst_norm(self.up(x))
        return x

class Pyramid2D(nn.Module):
    def __init__(self, ch_in=3, ch_step=8):
        super(Pyramid2D, self).__init__()

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
        self.last_conv = nn.Conv2d(6*ch_step, 3, 1, padding=0, bias=True)

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
