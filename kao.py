import kaolin as kal
from kaolin.graphics import NeuralMeshRenderer as Renderer 
from kaolin.graphics.nmr.util import get_points_from_angles

import torch
import torch.nn as nn

import numpy as np

import utils

import os
import argparse
import tqdm
from skimage.io import imread


MESH_DIR = './data/3d-models/chairs'
MESH_FILE = 'rocket.obj'

STYLE_DIR = './data/images/selected_styles'
STYLE_FILE = 'chair-2.jpg'

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

    def __init__(self,mesh_path, image_path, args):
        super(Model,self).__init__()
        self.args = args

        mesh = kal.rep.TriangleMesh.from_obj(mesh_path)
        mesh.cuda()

        vertices =  normalize_vertices(mesh.vertices).unsqueeze(0)
        faces = mesh.faces.unsqueeze(0)

        self.register_buffer('vertices',vertices)
        self.register_buffer('faces',faces)

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


    
    def forward(self):

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
        loss=1


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

    model = Model(args.mesh,args.image,args)
    model.cuda()

    model()

    # # Load mesh
    # mesh = kal.rep.TriangleMesh.from_obj(mesh_path)
    # mesh.cuda()

    # # Visualize mesh
    # # kal.visualize.show_mesh(mesh)

    # vertices = normalize_vertices(mesh.vertices).unsqueeze(0)
    # faces = mesh.faces.unsqueeze(0)

    # # Load style furniture image
    # style_path = os.path.join(STYLE_DIR,STYLE_FILE)
    # style = utils.image_to_tensor(utils.load_image(style_path))

    # # Initialize textures
    # textures = torch.zeros(
    #     1,faces.shape[1],4,4,4,
    #     3,dtype=torch.float32,
    #     device='cuda:0'
    # )
    

    # # Setup renderer
    # renderer = Renderer(camera_mode='look_at')
    # renderer.light_intensity_directional = 0.0
    # renderer.light_intensity_ambient = 1.0
    
    # cam_distance = 2.732
    # elevation =  0.0
    # renderer.eye = get_points_from_angles(
    #     cam_distance,
    #     elevation,
    #     np.random.uniform(0,360)
    # )

    # rendered_img,_,_ = renderer(
    #     vertices,
    #     faces,
    # )

    # Mask out style

    # Setup renderer

    # Get texture of mesh

    # Style transfer from style furn to mesh texture

    # Output texture is new texture








if __name__ == '__main__':
    main()
    
