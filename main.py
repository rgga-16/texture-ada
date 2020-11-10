import torch


import style_transfer as st
import utils
import args

from kao import MeshRendererModel,texture_transfer_gatys
from args import parse_arguments

from defaults import DEFAULTS as D


def main():
    print("Main Driver")

    args = parse_arguments()
    
    # Load style furniture image
    style = utils.image_to_tensor(utils.load_image(args.style)).detach()
    
    mesh = utils.load_mesh(args.mesh)
    
    # Create model for 3D texture transfer
    render_model = MeshRendererModel(mesh,style)
    
    texture_transfer_gatys(render_model,style)



if __name__ == "__main__":
    main()
    






   
