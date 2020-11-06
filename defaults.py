# Module to place all default variables in
import torch   
import torchvision.transforms as transforms


from enum import Enum
import pathlib as p



class DEFAULTS(Enum):
    DEVICE_ID = "cuda" if torch.cuda.is_available() else "cpu"
    DEVICE_ = torch.device(DEVICE_ID)

    IMSIZE = 512
    EPOCHS_ = 750

    NORM_MEAN = [0.485,0.456,0.406]
    NORM_STD = [0.229,0.224,0.225]

    CAM_DISTANCE_ = 2.732
    
    CONTENT_LAYERS = {
        '22' : 'relu4_2',
    }
    CL_WEIGHTS = {
        layer: 1.0 for layer in CONTENT_LAYERS.values()
    }

    STYLE_LAYERS = {
        '3': 'relu1_2',   # Style layers
        '8': 'relu2_2',
        '17' : 'relu3_4',
        '26' : 'relu4_4',
        '35' : 'relu5_4',
    }
    SL_WEIGHTS = {
        layer: 0.2 for layer in STYLE_LAYERS.values()
    }

    MASK_DIR = 'data/images/masks'
    MASK_FILE = 'binary_mask.png'
    MASK_PATH_ = p.Path.cwd() / MASK_DIR / MASK_FILE

    STYLE_DIR = 'data/images/selected_styles'
    STYLE_FILE = 'chair-2_cropped_more.png'
    STYLE_PATH_ = p.Path.cwd() / STYLE_DIR / STYLE_FILE

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

    MESH_DATA_DIR = 'data/3d-models/ShapeNet/samples'
    MESH_DIR = CHAIRS[1]
    MESH_FILE = 'model.obj'
    TEXTURE_FILE = 'model.mtl'
    MESH_PATH_ = p.Path.cwd() / MESH_DATA_DIR / MESH_DIR / MESH_FILE
    TEXTURE_PATH_ = p.Path.cwd() / MESH_DATA_DIR / MESH_DIR / TEXTURE_FILE
    TEXTURE_SIZE_ = 8
    

    def get(self):
        return self.value
    
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

