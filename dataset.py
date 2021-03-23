import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

import os
from helpers import utils
from defaults import DEFAULTS as D


class UV_Style_Paired_Dataset(Dataset):

    def __init__(self, uv_dir,style_dir, uv_sizes, style_size) -> None:
        super().__init__()
        
        self.uv_dir = uv_dir
        self.style_dir = style_dir 
        self.uv_sizes = uv_sizes 
        self.style_size = style_size 
        self.uv_files = []
        self.style_files = []

        # Temporary code. for testing purposes
        ######
        # self.uv_file = 'right_arm_uv.png'
        # self.style_file = 'chair-3_tiled.png'
        # self.uv_files.append(os.path.join(self.uv_dir,self.uv_file))
        # self.style_files.append(os.path.join(self.style_dir,self.style_file))
        ######

        # lounge sofa
        uv_map_style_pairings = {
            'left_arm_uv.png':'chair-3_tiled.png',
            'right_arm_uv.png':'chair-3_tiled.png',
            'left_backseat_uv.png':'cobonpue-17_tiled.png',
            'mid_backseat_uv.png':'chair-2_tiled.png',
            'right_backseat_uv.png':'cobonpue-17_tiled.png',
            'left_base_uv.png':'cobonpue-80_tiled.png',
            'right_base_uv.png':'cobonpue-80_tiled.png',
            'left_seat_uv.png':'cobonpue-99_tiled.png',
            'mid_seat_uv.png':'chair-2_tiled.png',
            'right_seat_uv.png':'cobonpue-99_tiled.png',
        }

        for k,v in uv_map_style_pairings.items():
            self.uv_files.append(os.path.join(self.uv_dir,k))
            self.style_files.append(os.path.join(self.style_dir,v))

        # for file in os.listdir(self.uv_dir):
        #     self.uv_files.append(os.path.join(self.uv_dir,file))
        # for file in os.listdir(self.style_dir):
        #     self.style_files.append(os.path.join(self.style_dir,file))

        assert len(self.style_files) == len(self.uv_files)

    
    def __len__(self):
        return len(self.style_files)

    def __getitem__(self, index):
        uv_map_set = []
        for uv_size in self.uv_sizes:
            uv_map_set.append(utils.image_to_tensor(utils.load_image(self.uv_files[index]),image_size = uv_size).detach())
        
        style = utils.image_to_tensor(utils.load_image(self.style_files[index]),image_size=self.style_size).detach()
        style = style[:3,...]
        sample = {'style': style, 'uvs':uv_map_set}
        
        return sample
 