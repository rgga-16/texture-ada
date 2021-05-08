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

def evaluate_texture(model:BaseModel,test_loader):
    args = args_.parse_arguments()

    assert isinstance(model,BaseModel)
    model.eval()

    w = args.uv_test_sizes[0]
    running_dist=0.0
    running_loss=0.0
    for i,texture in enumerate(test_loader):
        model.set_input(texture)
        with torch.set_grad_enabled(False):
            output=model.forward()
            loss,wdist=model.get_losses()
            running_dist+=wdist*texture.shape[0]
            running_loss+=loss*texture.shape[0]
    
    eval_wdist = running_dist/test_loader.dataset.__len__()
    eval_loss = running_loss/test_loader.dataset.__len__()
    return eval_loss,eval_wdist

def predict_texture(model:BaseModel,texture,output_path,mask=None):
    args = args_.parse_arguments()

    assert isinstance(model,BaseModel)

    model.eval()
    
    w = args.uv_test_sizes[0]
    texture = texture.expand(1,-1,-1,-1).clone().detach()
    
    with torch.no_grad():
        model.set_input(texture)
        output = model.forward()
        loss,wdist = model.get_losses()

    output_image = image_utils.tensor_to_image(output,image_size=args.output_size)

    
    if mask is not None:
        mask = mask.resize(output_image.size) if mask.size != output_image.size else ...
        output_image.putalpha(mask)
    output_image.save(output_path,'PNG')
    print('Saving image as {}'.format(output_path))
    return loss,wdist




