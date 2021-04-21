import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

import os, json 
from helpers import image_utils
from defaults import DEFAULTS as D


class UV_Style_Paired_Dataset(Dataset):

    def __init__(self, uv_dir,style_dir, uv_sizes, style_size, uv_style_pairs:dict) -> None:
        super().__init__()

        self.uv_dir = uv_dir
        self.style_dir = style_dir 
        self.uv_sizes = uv_sizes 
        self.style_size = style_size 
        self.uv_files = []
        self.style_files = []
        
        for k,v in uv_style_pairs.items():
            self.uv_files.append(os.path.join(self.uv_dir,k))
            self.style_files.append(os.path.join(self.style_dir,v))

        assert len(self.style_files) == len(self.uv_files)
    
    def __len__(self):
        return len(self.style_files)

    def __getitem__(self, index):
        
        style_im = self.style_files[index]
        style = image_utils.image_to_tensor(image_utils.load_image(style_im),image_size=self.style_size).detach()
        style = style[:3,...]

        uv_map_set = []
        for uv_size in self.uv_sizes:
            uv_map_set.append(image_utils.image_to_tensor(image_utils.load_image(self.uv_files[index]),image_size = uv_size).detach())

        sample = {'style': style, 'uvs':uv_map_set}
        
        return sample



