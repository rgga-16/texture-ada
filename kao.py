import kaolin as kal
from kaolin.graphics import NeuralMeshRenderer as Renderer 
from kaolin.graphics.nmr.util import get_points_from_angles
from kaolin.datasets.modelnet import ModelNet


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

import copy
import os
import argparse
from tqdm import tqdm
from skimage.io import imread

from defaults import DEFAULTS as D

from args import parse_arguments

class Model(nn.Module):

    def content_loss(self,output_mesh,content_mesh):
        return

    def __init__(self,mesh, style, args):
        super(Model,self).__init__()
        self.args = args

        self.mesh = mesh

        vertices =  normalize_vertices(self.mesh.vertices).unsqueeze(0)
        faces = self.mesh.faces.unsqueeze(0)

        self.register_buffer('vertices',vertices)
        self.register_buffer('faces',faces)
        self.register_buffer('style',style)

        # Initialize textures
        textures = torch.ones(
            1,self.faces.shape[1],
            self.args.texture_size,self.args.texture_size,self.args.texture_size,
            3,dtype=torch.float32,
            device=D.DEVICE()
        )

        self.textures = nn.Parameter(textures)

        # Setup renderer
        renderer = Renderer(image_size=D.IMSIZE.get(),camera_mode='look_at')
        renderer.light_intensity_directional = 0.0
        renderer.light_intensity_ambient = 2.0
        self.renderer = renderer

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

def normalize_vertices(vertices):
    """
    Normalize mesh vertices into a unit cube centered at zero.
    """
    vertices = vertices - vertices.min(0)[0][None, :]
    vertices /= torch.abs(vertices).max()
    vertices *= 2
    vertices -= vertices.max(0)[0][None, :] / 2
    return vertices

def to_device(input):
    input.to(device)
    return input


def main():

    args = parse_arguments()

    transform = kal.transforms.Compose([
        to_device,
        kal.transforms.NormalizeMesh()
    ])

    dataset = ModelNet(root='./data/3d-models/ModelNet10',
                        split='train',
                        transform=transform,
                        categories=['chair'])

    mesh = next(iter(dataset)).data

    # Load style furniture image
    
    style = utils.image_to_tensor(utils.load_image(args.image)).to(device).detach()

    
    # Create model for 3D texture transfer
    render_model = Model(mesh,style,args).to(device)

    optim = torch.optim.Adam(render_model.parameters(),lr=1e-2,betas=(0.5,0.999))

    progress_bar = tqdm(range(args.epochs))
    for i in progress_bar:
        optim.zero_grad()
        diff = render_model()
        progress_bar.set_description('EPOCH {} | LOSS: {}'.format(i+1,diff.item()))
        diff.backward()
        optim.step()

    for i in tqdm(range(0,360,45)):
        utils.tensor_to_image(render_model.render_image(azimuth=i,elevation=-100)).save('outputs/renders/front_{}.png'.format(i))
        utils.tensor_to_image(render_model.render_image(azimuth=i,elevation=100)).save('outputs/renders/back_{}.png'.format(i))

    # initial_img = rendered_img.clone().detach()

    # # Style transfer from style furn to mesh texture
    # final = st.style_transfer_gatys2(st_model,rendered_img, style,initial_img)
    


if __name__ == '__main__':
    device = D.DEVICE()
    main()
    
