import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

import os, json, random 
from helpers import image_utils, model_utils
from defaults import DEFAULTS as D
import args as args_

class Pix3D(Dataset):
    def __init__(self, n_points,categories:list=[],json_path='./data/3d-models/Pix3D/pix3d.json',lower_size=None) -> None:
        super().__init__()

        self.root_dir = os.path.dirname(json_path)
        self.data = json.load(open(json_path))
        self.n_points=n_points

        if len(categories)>0: 
            self.data = [d for d in self.data if d['category'].lower() in categories]
        
        if lower_size:
            self.data = random.sample(self.data,lower_size)

        print()
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        # args = args_.parse_arguments()
        sample = self.data[index]

        model_path = os.path.join(self.root_dir,sample['model'])
        image_path = os.path.join(self.root_dir,sample['img'])
        mask_path = os.path.join(self.root_dir,sample['mask'])

        # Preprocess model
        verts,faces,_ = model_utils.load_mesh(model_path)
        pointcloud = model_utils.mesh_to_pointcloud(verts,faces,self.n_points)

        img = image_utils.load_image(image_path)
        mask = image_utils.load_image(mask_path,mode='L')
        masked_img = img 
        masked_img.putalpha(mask)

        # Preprocess image
        image_tensor = image_utils.image_to_tensor(masked_img,phase='test',image_size=256)[:3,...]
        mask_tensor = image_utils.image_to_tensor(mask,phase='test',image_size=256,normalize=False)
        return pointcloud.squeeze(),image_tensor.squeeze(),mask_tensor.squeeze()
        

class Pix3D_Paired(Pix3D):
    def __init__(self, n_points,categories:list=[],json_path='./data/3d-models/Pix3D/pix3d.json',lower_size=None) -> None:
        super().__init__(n_points,categories,json_path=json_path,lower_size=lower_size)
        
        
        half = len(self.data)//2
        self.set1 = self.data[:half]
        self.set2 = self.data[half:]

        assert len(self.set1) == len(self.set2)
    
    def __len__(self):
        return len(self.set1)
    
    def __getitem__(self, index):
        sample1 = self.set1[index]
        sample2 = self.set2[index]

        pointclouds=[]
        images=[]
        masks=[]
        for sample in [sample1,sample2]:
            model_path = os.path.join(self.root_dir,sample['model'])
            image_path = os.path.join(self.root_dir,sample['img'])
            mask_path = os.path.join(self.root_dir,sample['mask'])

            # Preprocess model
            verts,faces,_ = model_utils.load_mesh(model_path)
            pointcloud = model_utils.mesh_to_pointcloud(verts,faces,self.n_points)

            img = image_utils.load_image(image_path)
            mask = image_utils.load_image(mask_path,mode='L')
            masked_img = img 
            masked_img.putalpha(mask)

            # Preprocess image
            image_tensor = image_utils.image_to_tensor(masked_img,phase='test',image_size=256)[:3,...]
            mask_tensor = image_utils.image_to_tensor(mask,phase='test',image_size=256,normalize=False)
            masks.append(mask_tensor.squeeze())
            pointclouds.append(pointcloud.squeeze())
            images.append(image_tensor.squeeze())

        return pointclouds[0],images[0],masks[0],pointclouds[1],images[1],masks[1]

