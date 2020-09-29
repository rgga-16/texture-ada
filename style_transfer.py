import torch
import torchvision

from torchvision import models
from torchvision import transforms

import losses
import utils
import dextr.segment as seg

from PIL import Image

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

def interactive_style_transfer(model,content_path,style_paths,device,IMSIZE=256,EPOCHS=1000):
    save_path = content_path
    i=0
    for style_path in style_paths:

        _,mask_path = seg.segment_points(save_path,device=device)
        mask_img = utils.load_image(mask_path)
        mask = utils.image_to_tensor(mask_img,image_size=IMSIZE).to(device)

        _,style_mask_path = seg.segment_points(style_path,device=device)
        style_mask_img = utils.load_image(style_mask_path)
        style_mask = utils.image_to_tensor(style_mask_img,image_size=IMSIZE).to(device)

        print("Mask shape: {}".format(mask.shape))
        print("Style mask shape: {}".format(style_mask.shape))

        content_img = utils.load_image(save_path)
        style_img = utils.load_image(style_path)

        content = utils.image_to_tensor(content_img,image_size=IMSIZE).to(device)
        style = utils.image_to_tensor(style_img,image_size=IMSIZE).to(device)

        content_clone = content.clone().detach()

        content = content * mask

        style = style * style_mask
        
        print("Mask shape: {}".format(mask.shape))
        print("content shape: {}".format(content.shape))
        print("style shape: {}".format(style.shape))

        # setup normalization mean and std
        normalization_mean = torch.tensor([0.485,0.456,0.406]).to(device)
        normalization_std = torch.tensor([0.229,0.224,0.225]).to(device)

        initial = content.clone()

        output = style_transfer_gatys(model,normalization_mean,normalization_std,content,style,initial,EPOCHS=EPOCHS)

        save_path = 'outputs/stylized_output_{}.png'.format(i+1)
        i+=1
        final = (output * mask) + (content_clone * (1-mask))
        final_img = utils.tensor_to_image(final)
        final_img.save(save_path)
