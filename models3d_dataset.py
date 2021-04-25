import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

import os, json 
from helpers import image_utils, model_utils
from defaults import DEFAULTS as D
from args import args


class Pix3D(Dataset):
    def __init__(self, n_points,category:str=None,json_path='./data/3d-models/Pix3D/pix3d.json') -> None:
        super().__init__()

        self.root_dir = os.path.dirname(json_path)
        self.data = json.load(open(json_path))
        self.n_points=n_points

        if category is not None: 
            self.data = [d for d in self.data if d['category'].lower()==category.lower()]
        print()
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        sample = self.data[index]

        model_path = os.path.join(self.root_dir,sample['model'])
        image_path = os.path.join(self.root_dir,sample['img'])
        mask_path = os.path.join(self.root_dir,sample['mask'])
        # rot_mat = torch.tensor(sample['rot_mat']).expand(1,-1,-1)

        # Preprocess model
        verts,faces,_ = model_utils.load_mesh(model_path)
        pointcloud = model_utils.mesh_to_pointcloud(verts,faces,self.n_points)

        # Preprocess image
        image = image_utils.image_to_tensor(image_utils.load_image(image_path),image_size=args.style_size)

        return pointcloud.squeeze(),image.squeeze()
        