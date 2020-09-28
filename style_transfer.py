import torch
import torchvision

from torchvision import models
from torchvision import transforms

import losses
import utils

def get_optimizer(output_img):
    optim = torch.optim.Adam([output_img.requires_grad_()],lr=1e-2)
    return optim


style_layers_default = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']

def style_transfer_gatys(cnn,normalization_mean, normalization_std,
                    content_img, style_img, output_img, EPOCHS=500,
                    style_weight=1e6,content_weight=1,mask=None,style_layers=style_layers_default):
    
    model, style_losses, content_losses = losses.get_model_and_losses(cnn,normalization_mean,
                                                                    normalization_std,
                                                                    style_img,content_img,mask=mask,style_layers=style_layers)

    optimizer = get_optimizer(output_img)

    run = [0] 
    while run[0] <= EPOCHS:

        def closure():
            output_img.data.clamp_(0,1)
            optimizer.zero_grad()

            model(output_img)

            style_loss = 0
            content_loss = 0

            for sl in style_losses:
                style_loss += sl.loss
            
            for cl in content_losses:
                content_loss += cl.loss

            style_loss *= style_weight
            content_loss *= content_weight
            loss = style_loss + content_loss
            loss.backward()

            run[0] += 1
            if(run[0] % 50 == 0):
                print('Iter {} | Total Loss: {:4f} | Style Loss: {:4f} | Content Loss: {:4f}'.format(run[0],loss.item(),content_loss.item(),style_loss.item()))
        
        optimizer.step(closure)
    
    output_img.data.clamp_(0,1)
    return output_img

# def region_style_transfer(content)