import kaolin as kal
from kaolin.graphics import NeuralMeshRenderer as Renderer 
from kaolin.graphics.nmr.util import get_points_from_angles
from kaolin.datasets.modelnet import ModelNet 
from kaolin.datasets.shapenet import ShapeNet


import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.model_zoo import load_url
from torchvision import models

import numpy as np

import utils
import losses
import dextr.segment as s
import style_transfer as st
from models import VGG19

import copy
import os
import argparse
from tqdm import tqdm
from skimage.io import imread

from defaults import DEFAULTS as D

from args import parse_arguments

class MeshRendererModel(nn.Module):

    def __init__(self,mesh, style,
                textures_=None,texture_size=D.TEXTURE_SIZE(),
                camera_distance = D.CAM_DISTANCE(),
                device=D.DEVICE()):
        super(MeshRendererModel,self).__init__()


        self.mesh = mesh
        vertices =  self.normalize_vertices(self.mesh.vertices).unsqueeze(0)
        faces = self.mesh.faces.unsqueeze(0)
        self.register_buffer('vertices',vertices)
        self.register_buffer('faces',faces)
        self.register_buffer('style',style)

        if(textures_ is not None):
            textures = textures_
        else:
            # Initialize textures
            textures = torch.ones(
                1,self.faces.shape[1],
                texture_size,texture_size,texture_size,
                3,dtype=torch.float32,
                device=device
            )
        
        self.camera_distance=camera_distance

        self.textures = nn.Parameter(textures)

        # Setup renderer
        renderer = Renderer(image_size=D.IMSIZE.get(),camera_mode='look_at')
        renderer.light_intensity_directional = 0.0
        renderer.light_intensity_ambient = 2.0
        self.renderer = renderer

        self.transform = kal.transforms.Compose([
            self.to_device,
            kal.transforms.NormalizeMesh(),
        ])

        self.device=device
        self.to(device)


    """
    Casts input to the device
    """
    def to_device(self,input):
        input.to(self.device)
        return input
    
    

    def render_image(self, azimuth=None, elevation=None):

        if azimuth is not None:
            angle = azimuth
        else:
            angle= np.random.uniform(0,360)

        if elevation is not None:
            elev = elevation 
        else:
            elev = np.random.uniform(-100,100)

        self.renderer.eye = get_points_from_angles(
            self.args.camera_distance,
            elev,
            angle
        )

        image, _, _ = self.renderer(
            self.vertices,
            self.faces,
            torch.tanh(self.textures)
        )
        return image

    
    def forward(self,iters=1):
        total_loss = 0
        for i in range(iters):
            image = self.render_image()
            loss = torch.sum((image-self.style)**2)
            total_loss+=loss
        return total_loss


def texture_transfer_gatys(render_model,style,
                            descriptor=VGG19(),EPOCHS=D.EPOCHS(),
                            style_layers = D.STYLE_LAYERS.get(),
                            style_weight=1e6,
                            s_layer_weights=D.SL_WEIGHTS.get()):
    
    optimizer = torch.optim.SGD(render_model.parameters(),lr=1e-2)

    progress = tqdm(range(EPOCHS))
    mse_loss = nn.MSELoss()


    style_gram = st.get_features(descriptor,style,style_layers=style_layers)

    for i in progress:

        optimizer.zero_grad()
        loss=0
        rendered_img = render_model.render_image()
        rendered_img.data.clamp_(0,1)

        rendered_gram = st.get_features(descriptor,rendered_img,style_layers=style_layers)

        for layer in style_layers.values():
            diff = torch.sum((rendered_gram[layer]-style_gram[layer])**2)
            diff *= s_layer_weights[layer]
            loss+=diff
        
        loss.backward()
        optimizer.step()
        progress.set_description('EPOCH {} | LOSS: {}'.format(i+1,loss.item()))
    
    rendered_imgs = []
    for i in tqdm(range(0,360,15)):
        image = render_model.render_image(azimuth=i,elevation=15)
        image.data.clamp_(0,1)
        rendered_imgs.append(utils.tensor_to_image(image))
    
    utils.save_gif(rendered_imgs,filename='./outputs/renders/rendered_model.gif')



def main():

    args = parse_arguments()
    
    # Load style furniture image
    style = utils.image_to_tensor(utils.load_image(args.style)).detach()
    
    mesh = utils.load_mesh(args.mesh)

    kal.visualize.show(mesh)
    
    # # Create model for 3D texture transfer
    # render_model = MeshRendererModel(mesh,style)
    
    # texture_transfer_gatys(render_model,style)






if __name__ == '__main__':
    main()
    
