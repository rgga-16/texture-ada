# Module to place all default variables in
import torch   
import torchvision.transforms as transforms

import numpy as np 
import random 


from enum import Enum
import pathlib as p
import datetime


class DEFAULTS(Enum):

    DATE_ = datetime.datetime.today().strftime('%m-%d-%y %H-%M-%S')


    DEVICE_ID = "cuda" if torch.cuda.is_available() else "cpu"
    DEVICE_ = torch.device(DEVICE_ID)

    LR_=1e-4

    IMSIZE = 256
    EPOCHS_ = 5000

    NORM_MEAN = [0.485,0.456,0.406]
    NORM_STD = [0.229,0.224,0.225]

    CAM_DISTANCE_ = 2.732
    
    CONTENT_LAYERS = {
        '22' : 'relu4_2',
    }
    CL_WEIGHTS = {
        layer: 1.0 for layer in CONTENT_LAYERS.values()
    }

    # STYLE_LAYERS = {
    #     '3': 'relu1_2',   # Style layers
    #     '8': 'relu2_2',
    #     '13' : 'relu3_4',
    #     '26' : 'relu4_4',
    # }

    STYLE_LAYERS = {
        '1': 'relu1_1',   # Style layers
        '6': 'relu2_1',
        '11' : 'relu3_1',
        '20' : 'relu4_1',
    }
    SL_WEIGHTS = {
        layer: 0.25 for layer in STYLE_LAYERS.values()
    }

    MASK_DIR = 'data/images/masks'
    MASK_FILE = 'binary_mask.png'
    MASK_PATH_ = p.Path.cwd() / MASK_DIR / MASK_FILE

    STYLE_DIR = 'inputs/style_images/tiled'
    STYLE_FILE = 'chair-3_tiled.png'
    STYLE_PATH_ = p.Path.cwd() / STYLE_DIR / STYLE_FILE

    MODEL_DIR = './models'

    CHAIRS = [
        'armchair sofa', # 0 no texture
        'dining table chair',# 1 has texture
        'office chair', # 2:office swivel chair no texture
        'steel chair', # 3:steel chair no texture
        'armchair', # 4:armchair has texture
        'sofa chair no arms', # 5:sofa chair no arms has texture
        'wooden armchair pinoy', # 6:armchair filipino no texture
        'armchair sofa 2', # 7:armchair sofa 2 no texture
        'office chair 2', # 8:office swivel chair with headrest no texture
        'armchair sofa 3', # 9:armchair sofa 3 no texture
    ]

    MESHES_DIR = 'inputs/shape_samples'
    MESH_DIR = CHAIRS[0]
    MESH_FILE = 'backseat.obj'
    
    TEXTURE_FILE = 'model.mtl'
    MESH_PATH_ = p.Path.cwd() / MESHES_DIR / MESH_DIR / MESH_FILE
    TEXTURE_PATH_ = p.Path.cwd() / MESHES_DIR / MESH_DIR / TEXTURE_FILE
    TEXTURE_SIZE_ = 8
    

    def get(self):
        return self.value

    @classmethod
    def LR(cls):
        return cls.LR_.value
    
    @classmethod
    def DATE(cls):
        return cls.DATE_.value
    
    @classmethod
    def DEVICE(cls):
        return cls.DEVICE_.value
    
    @classmethod
    def MESH_PATH(cls):
        return cls.MESH_PATH_.value 
    
    @classmethod
    def TEXTURE_PATH(cls):
        return cls.TEXTURE_PATH_.value
    
    @classmethod
    def TEXTURE_SIZE(cls):
        return cls.TEXTURE_SIZE_.value
    
    @classmethod
    def CAM_DISTANCE(cls):
        return cls.CAM_DISTANCE_.value

    @classmethod 
    def STYLE_PATH(cls):
        return cls.STYLE_PATH_.value

    @classmethod
    def MASK_PATH(cls):
        return cls.MASK_PATH_.value
    
    @classmethod
    def EPOCHS(cls):
        return cls.EPOCHS_.value

