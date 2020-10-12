# Module to place all default variables in
import torch   
import torchvision.transforms as transforms
import utils


from enum import Enum
import pathlib as p



class DEFAULTS(Enum):
    DEVICE_ID = "cuda"
    DEVICE_ = torch.device(DEVICE_ID)
    IMSIZE = 256

    NORM_MEAN = [0.485,0.456,0.406]
    NORM_STD = [0.229,0.224,0.225]
    
    CONTENT_LAYERS = {
        '22' : 'relu4_2',
    }
    CL_WEIGHTS = {
        layer: 1.0 for layer in CONTENT_LAYERS.values()
    }

    STYLE_LAYERS = {
        '3': 'relu1_2',   # Style layers
        '8': 'relu2_2',
        '17' : 'relu3_3',
        '26' : 'relu4_3',
        '35' : 'relu5_3',
    }
    SL_WEIGHTS = {
        layer: 0.2 for layer in STYLE_LAYERS.values()
    }

    MESH_DIR = 'data/3d-models/chairs'
    MESH_FILE = 'rocket.obj'
    MESH_PATH = p.Path.cwd() / MESH_DIR / MESH_FILE

    STYLE_DIR = 'data/images/selected_styles'
    STYLE_FILE = 'starry.jpg'
    STYLE_PATH = p.Path.cwd() / STYLE_DIR / STYLE_FILE

    def get(self):
        return self.value
    
    @classmethod
    def DEVICE(cls):
        return cls.DEVICE_.value

