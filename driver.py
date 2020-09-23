
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
# import blender as b

import argparse

import torch.nn.functional as F

from torchvision import models

import os

IMSIZE=256

datapath = './data'
datatype='images'
furniture='chairs'

generic='generic/sofa-2.jpeg'

style='selected_styles'

des_furn = [
    'cobonpue/chair-2.jpg',
    'budji_layug/chair-1.jpg',
    'vito_selma/chair-3.jpg'
]

if __name__ == "__main__":
    print("Main Driver")

    content_path = os.path.join(datapath,datatype,furniture,generic)

    style_dir = os.path.join(datapath,datatype,style)

    style_paths = [os.path.join(style_dir,fil) for fil in os.listdir(style_dir)]
    
    device = utils.setup_device(use_gpu = True)
    model = models.vgg19(pretrained=True).features.to(device).eval()

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

        output = st.run_style_transfer(model,normalization_mean,normalization_std,content,style,initial,EPOCHS=1000)

        save_path = 'outputs/stylized_output_{}.png'.format(i+1)
        i+=1
        final = (output * mask) + (content_clone * (1-mask))
        final_img = utils.tensor_to_image(final)
        final_img.save(save_path)


