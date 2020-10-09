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

def get_model_and_losses(cnn, style_img, content_img,
                        normalization_mean=normalization_mean_default, 
                        normalization_std=normalization_std_default, 
                        content_layers = content_layers_default,
                        style_layers=style_layers_default, mask=None):
    
    cnn = copy.deepcopy(cnn)
    
    # normalization
    normalization = Normalization(normalization_mean,normalization_std)

    content_losses = []
    style_losses = []

    # model = nn.Sequential(normalization)
    model = nn.Sequential()
    
    i=0 #Increment everytime we see a conv layer
    
    # Rename layers and add to newly created model
    for layer in cnn.children():

        if isinstance(layer,nn.Conv2d):
            i+=1
            name = 'conv_{}'.format(i)

        elif isinstance(layer,nn.ReLU):
            name ='relu_{}'.format(i)
            layer=nn.ReLU(inplace=False)

        elif isinstance(layer,nn.MaxPool2d):
            name = 'pool_{}'.format(i)
        elif isinstance(layer,nn.BatchNorm2d):
            name = 'batchnorm_{}'.format(i)
        else:
            raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))

        model.add_module(name,layer)
        
        # Retrieve content losses for each content layer
        if name in content_layers and content_img is not None:
            output = model(content_img).detach()
            content_loss = ContentLoss(output)
            model.add_module('content_loss_{}'.format(i),content_loss)
            content_losses.append(content_loss)

        # Retrieve style losses for each style layer
        if name in style_layers and style_img is not None:
            output_features = model(style_img).detach()
            style_loss = StyleLoss(output_features)
            model.add_module('style_loss_{}'.format(i),style_loss)
            style_losses.append(style_loss)
    
    # Remove last layer of content and style losses
    for i in range(len(model)-1,-1,-1):
        if isinstance(model[i],ContentLoss) or isinstance(model[i],StyleLoss):
            break

    model = model [:(i+1)]
    return model,style_losses,content_losses