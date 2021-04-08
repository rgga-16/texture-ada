import torch
from torch.utils.data import DataLoader

from args import args
from dataset import UV_Style_Paired_Dataset
from defaults import DEFAULTS as D
from helpers import logger, utils 
from models import VGG19, Pyramid2D
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
    
    uvs = input[:-1]
    style=input[-1][:3,...].unsqueeze(0).detach()
    for uv in uvs:
        _,_,w = uv.shape
        input_sizes = [w//2,w//4,w//8,w//16,w//32]
        inputs = [uv[:3,...].unsqueeze(0).detach()]
        # inputs.extend([torch.rand(1,3,sz,sz,device=D.DEVICE()) for sz in input_sizes])
        inputs.extend([F.interpolate(style,sz,mode='nearest') for sz in input_sizes])

        with torch.no_grad():
            y = generator(inputs)

        output_path_ = '{}_{}.png'.format(output_path,w)
        utils.tensor_to_image(y,image_size=args.output_size).save(output_path_)
        print('Saving image as {}'.format(output_path_))