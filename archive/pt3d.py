
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.model_zoo import load_url
from torchvision import models

import numpy as np
import matplotlib.pyplot as plt

import pytorch3d as p3d 
from pytorch3d.io import load_obj, load_objs_as_meshes

# Data structures and functions for rendering
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras, 
    PointLights, 
    DirectionalLights, 
    Materials, 
    RasterizationSettings, 
    MeshRenderer, 
    MeshRasterizer,  
    SoftPhongShader,
    TexturesUV
)

import utils
# import losses
# import style_transfer as st
# from models import VGG19

# import copy
# import os
# from tqdm import tqdm
# from skimage.io import imread

from defaults import DEFAULTS as D

from args import parse_arguments

def get_renderer(dist=1.0,elev=0,azim=0):
    # Setup camera
    R,T = look_at_view_transform(dist=dist,elev=elev,azim=azim)
    cameras = FoVPerspectiveCameras(device=D.DEVICE(),R=R,T=T)

    # Setup rasterizer
    raster_settings = RasterizationSettings(
        image_size=D.IMSIZE.get(),
        blur_radius=0.0,
        faces_per_pixel=1
    )

    # Place a point light in front of the object
    lights=PointLights(device=D.DEVICE(),location=[[0.0,0.0,-3.0]])

    # Setup a phong renderer by composing a rasterizer and shader.
    # The textured phong shader will interpolate the texture uv coordinates 
    # for each vertex, sample from a texture image and apply the Phong lighting model
    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(
            cameras=cameras,
            raster_settings=raster_settings
        ),
        shader=SoftPhongShader(
            device=D.DEVICE(),
            cameras=cameras,
            lights=lights
        )
    )

    return renderer


if __name__ == '__main__':
    # args = parse_arguments()

    W=D.IMSIZE.get()
    H=W

    verts,faces,aux= load_obj(str(D.MESH_PATH()),load_textures=True,device=D.DEVICE())

    faces_idx=faces.verts_idx

    verts_uvs = verts.unsqueeze(0)
    verts_uvs= verts_uvs[:,:,:2]

    
    renderer = get_renderer()
    initial_texture = TexturesUV(maps=torch.ones(1,H,W,3).to(D.DEVICE()),
                                faces_uvs=faces_idx.unsqueeze(0),
                                verts_uvs=verts_uvs)

    mesh = Meshes(verts=[verts],faces=[faces_idx],textures=initial_texture)
    rendering = renderer(mesh)
    
    plt.figure(figsize=(10, 10))
    plt.imshow(rendering[0, ..., :3].cpu().numpy())
    plt.grid("off")
    plt.axis("off")
    plt.show()