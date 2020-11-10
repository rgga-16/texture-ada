
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.model_zoo import load_url
from torchvision import models

import numpy as np

import pytorch3d as p3d 
from pytorch3d.io import load_obj, load_objs_as_meshes

# import losses
# import style_transfer as st
# from models import VGG19

# import copy
# import os
# from tqdm import tqdm
# from skimage.io import imread

from defaults import DEFAULTS as D

from args import parse_arguments


args = parse_arguments()

verts,faces,aux= load_obj([str(args.mesh)],load_textures=False,device=D.DEVICE())
print()