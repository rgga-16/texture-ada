import torch
from torch.utils.data import DataLoader
import torchvision
import args as args_

from dataset import UV_Style_Paired_Dataset
from defaults import DEFAULTS as D
from helpers import logger, image_utils 
from models.base_model import BaseModel
import style_transfer as st

import numpy as np
from torch.utils.data import DataLoader
import os, copy, time, datetime ,json, matplotlib.pyplot as plt

def test_texture(model:BaseModel,texture,model_path,output_path,mask=None):
    args = args_.parse_arguments()

    assert isinstance(model,BaseModel)

    model.net.load_state_dict(torch.load(model_path))
    model.eval()
    
    w = args.uv_test_sizes[0]
    texture = texture.expand(1,-1,-1,-1).clone().detach()
    
    with torch.no_grad():
        model.set_input(texture)
        output = model.forward()
        loss,wdist = model.get_losses()

    output_path_ = '{}_{}.png'.format(output_path,w)
    output_image = image_utils.tensor_to_image(output,image_size=args.output_size)

    # img_grid = torchvision.utils.make_grid(torch.cat((texture,output),dim=0),normalize=True)
    # plt.axis('off')
    # plt.imshow(img_grid.cpu().permute(1,2,0))
    # plt.savefig(f'{output_path}_compare.png')
    # plt.clf()
    
    if mask is not None:
        mask = mask.resize(output_image.size) if mask.size != output_image.size else ...
        output_image.putalpha(mask)
    output_image.save(output_path_,'PNG')
    print('Saving image as {}'.format(output_path_))
    return loss,wdist




