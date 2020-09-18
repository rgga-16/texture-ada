
import utils
import style_transfer as st

import blender

from PIL import Image
import imageio
import cv2

import numpy as np
import torch

import dextr.segment as seg
import style_transfer as st
import blender as b

import argparse

import torch.nn.functional as F

from torchvision import models

import imageio


if __name__ == "__main__":
    print("Main Driver")

    device = utils.setup_device(use_gpu = True)

    # Setup model. Use pretrained VGG-19
    model = models.vgg19(pretrained=True).features.to(device).eval()

    content_path = './data/images/chairs/generic/armchair.jpeg'
    style_path = './data/images/chairs/cobonpue/chair-2.jpg'
    mask_path = './data/images/masks/segmented_back.png'
    style_mask_path = './data/images/masks/style_seat_mask.png'

    # ## Get mask by segmenting the content image via user input
    # mask,_ = seg.segment_points(content_path,device=device)
    # # Convert mask from Numpy array to Torch tensor
    # mask = torch.from_numpy(mask)
    # w,h,c = mask.shape
    # # Make mask have dimensions b,c,w,h
    # mask = mask.view(c,w,h).unsqueeze(0).to(device)

    ## Or get mask by loading from path
    mask_img = utils.load_image(mask_path)
    mask = utils.image_to_tensor(mask_img).to(device)

    style_mask = utils.image_to_tensor(utils.load_image(style_mask_path)).to(device)
    
    # style_mask,_ = seg.segment_points(style_path,device=device)
    # # Convert mask from Numpy array to Torch tensor
    # style_mask = torch.from_numpy(style_mask)
    # w,h,c = style_mask.shape
    # # Make mask have dimensions b,c,w,h
    # style_mask = style_mask.view(c,w,h).unsqueeze(0).to(device)

    print("Mask shape: {}".format(mask.shape))
    print("Style mask shape: {}".format(style_mask.shape))

    content_img = utils.load_image(content_path)
    style_img = utils.load_image(style_path)

    content = utils.image_to_tensor(content_img).to(device)
    style = utils.image_to_tensor(style_img).to(device)

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

    output = st.run_style_transfer(model,normalization_mean,normalization_std,content,style,initial,EPOCHS=2000)

    save_path = 'outputs/stylized_output.png'
    final = (output * mask) + (content_clone * (1-mask))
    final_img = utils.tensor_to_image(final)
    final_img.save(save_path)














    # output_np = output.detach().cpu().numpy()
    # content_np = content.detach().cpu().numpy()
    # mask_np = mask.detach().cpu().numpy()

    # final_img = b.blend_images(content_np,output_np,mask_np)
    
    # imageio.imwrite('outputs/stylized_output.png',final_img)
    
    # output_img = utils.tensor_to_image(output)
    # save_path = 'outputs/stylized_output.png'
    # output_img.save(save_path)
    # print('Saved at {}'.format(save_path))


    




