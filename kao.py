import kaolin as kal
from kaolin.graphics import NeuralMeshRenderer as Renderer 
from kaolin.graphics.nmr.util import get_points_from_angles
from kaolin.datasets.modelnet import ModelNet


import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import numpy as np

import utils
import dextr.segment as s

import os
import argparse
import tqdm
from skimage.io import imread


MESH_DIR = './data/3d-models/chairs'
MESH_FILE = 'rocket.obj'

STYLE_DIR = './data/images/selected_styles'
STYLE_FILE = 'chair-2.jpg'

MASK_PATH = './binary_mask.png'

mesh_path = os.path.join(MESH_DIR,MESH_FILE)
style_path = os.path.join(STYLE_DIR,STYLE_FILE)

def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Texture Transfer Algorithm')

    parser.add_argument('--mesh', type=str, default=mesh_path,
                        help='Path to the mesh OBJ file')
    parser.add_argument('--image', type=str, default=style_path,
                        help='Path to the style image to transfer texture from')
    parser.add_argument('--output_path', type=str, default='outputs',
                        help='Path to the output directory')
    parser.add_argument('--epochs', type=int, default=300,
                        help='Number of epochs to optimize')
    parser.add_argument('--camera_distance', type=float, default=2.732,
                        help='Distance from camera to object center')
    parser.add_argument('--elevation', type=float, default=0,
                        help='Camera elevation')
    parser.add_argument('--texture_size', type=int, default=4,
                        help='Dimension of texture')

    return parser.parse_args()

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
        textures = torch.zeros(
            1,self.faces.shape[1],self.args.texture_size,self.args.texture_size,self.args.texture_size,
            3,dtype=torch.float32,
            device='cuda'
        )

        self.textures = nn.Parameter(textures)

        # Setup renderer
        renderer = Renderer(camera_mode='look_at')
        renderer.light_intensity_directional = 0.0
        renderer.light_intensity_ambient = 1.0
        self.renderer = renderer

    def render_image(self):
        self.renderer.eye = get_points_from_angles(
            self.args.camera_distance,
            self.args.elevation,
            np.random.uniform(0,360)
        )

        image, _, _ = self.renderer(
            self.vertices,
            self.faces,
            torch.tanh(self.textures)
        )
        return image

    
    def forward(self):

        image = self.render_image()
        
        loss = torch.sum((image-self.style)**2)
        return loss

def normalize_vertices(vertices):
    """
    Normalize mesh vertices into a unit cube centered at zero.
    """
    vertices = vertices - vertices.min(0)[0][None, :]
    vertices /= torch.abs(vertices).max()
    vertices *= 2
    vertices -= vertices.max(0)[0][None, :] / 2
    return vertices

def main():

    args = parse_arguments()
    device = utils.setup_device(use_gpu = True)

    dataset = ModelNet(root='./data/3d-models/ModelNet10',
                        split='train')

    train_data = DataLoader(dataset)
    
    for batch_id, (data,attrib) in enumerate(train_data):
        print(batch_id)

    # # Load mesh
    # mesh = kal.rep.TriangleMesh.from_obj(args.mesh)
    # mesh.cuda()

    # # Load style furniture image
    # style_path = os.path.join(STYLE_DIR,STYLE_FILE)
    # style = utils.image_to_tensor(utils.load_image(style_path)).to(device)

    # # Mask out style
    # # _,mask_path = s.segment_points(style_path)
    # mask = utils.image_to_tensor(utils.load_image(MASK_PATH)).to(device)

    # # style = style * mask
    # cropped_style = utils.tensor_to_image(style)

    # # Create style feature extractor model
    # state = load_url('https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',model_dir='./models')
    # vgg = models.vgg19(pretrained=False).eval().to(device)
    # vgg.load_state_dict(state)
    # model = vgg.features

    # # Create model for 3D texture transfer
    # model = Model(mesh,style,args).to(device)
    # model()

    # # Style transfer from style furn to mesh texture
    # loop = tqdm.tqdm(range(args.epochs))
    # optimizer = torch.optim.Adam([
    #     p for p in model.parameters() if p.requires_grad
    # ],lr=0.1,betas=(0.5,0.999))
    # azimuth = 0.0

    # initial_mesh = mesh.

    # for i in loop:
    #     optimizer.zero_grad()

    #     loss = model()

    #     loss.backward()
    #     optimizer.step()

    # # Output texture is new texture
    # final = model.render_image()
    # final_img = utils.tensor_to_image(final).save('final_rendered.png')





if __name__ == '__main__':
    main()
    
