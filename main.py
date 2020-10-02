import torch
from torchvision import models
from torch.utils.model_zoo import load_url


import argparse
import os


import style_transfer as st
import utils
# import mesh


IMSIZE=256

datapath = './data'
datatype='images'
furniture='chairs'

generic='generic/chair-1.jpg'

style='selected_styles'


def create_arg_parser():

    default_style_path = os.path.join(datapath,datatype,style)
    default_content_path = os.path.join(datapath,datatype,furniture,generic)

    parser = argparse.ArgumentParser()
    parser.add_argument('-c','--content_path',type=str,help='Path to content image',default=default_content_path)
    parser.add_argument('-cm','--content_model',type=str,help='Path to content model')
    parser.add_argument('-s','--style_path',type=str,help='Path to style image or directory (if using 2+ style images)', default=default_style_path)
    parser.add_argument('-imsize', '--image_size', type=int,default=256)
    parser.add_argument('-o','--output_path', type=str,help='Path of output image')
    

    return parser


if __name__ == "__main__":
    print("Main Driver")

    parser = create_arg_parser()
    args = parser.parse_args()

    style_paths = [os.path.join(args.style_path,fil) for fil in os.listdir(args.style_path)]
    content_path = args.content_path

    device = utils.setup_device(use_gpu = True)
    
    state = load_url('https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',model_dir='./models')
    vgg = models.vgg19(pretrained=False).eval().to(device)
    vgg.load_state_dict(state)
    model = vgg.features

    # Load Mesh
    # sphere = mesh.create_sphere_mesh(device=device)

    # Setup style layers

    # Setup content layers
    




   
