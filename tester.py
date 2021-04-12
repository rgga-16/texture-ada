import torch
from torch.utils.data import DataLoader

from args import args
from dataset import UV_Style_Paired_Dataset
from defaults import DEFAULTS as D
from helpers import logger, utils 
from models.texture_transfer_models import VGG19, Pyramid2D
import style_transfer as st

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms


import os, copy, time, datetime ,json

def test(generator,input,gen_path,output_path):
    generator.eval()
    generator.cuda(device=D.DEVICE())

    generator.load_state_dict(torch.load(gen_path))
    
    # uvs = input[:-1]
    # style=input[-1][:3,...].unsqueeze(0).detach()
    uvs=input[:-1]
    style=input[-1]
    for uv in uvs:
        _,_,w = uv.shape
        input_sizes = [w//2,w//4,w//8,w//16,w//32]
        # inputs = [uv[:3,...].unsqueeze(0).detach()]
        # inputs.extend([torch.rand(1,3,sz,sz,device=D.DEVICE()) for sz in input_sizes])
        # inputs.extend([F.interpolate(style,sz,mode='nearest') for sz in input_sizes])
        inputs = uv[:3,...].unsqueeze(0).clone().detach()
        uv_mask = uv[3,...].expand(1,1,-1,-1).clone().detach()
        input_style = style[:3,...].unsqueeze(0).clone().detach()

        with torch.no_grad():
            output = generator(inputs,input_style)

        output_path_ = '{}_{}.png'.format(output_path,w)
        output_image = utils.tensor_to_image(output,image_size=args.output_size)
        mask = utils.tensor_to_image(uv_mask,image_size=args.output_size,denorm=False)
        output_image.putalpha(mask)
        output_image.save(output_path_,'PNG')
        print('Saving image as {}'.format(output_path_))