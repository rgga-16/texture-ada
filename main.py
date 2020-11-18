import torch


import style_transfer as st
import utils
import args

from args import parse_arguments
import os

from dextr.segment import segment_points

from defaults import DEFAULTS as D


def main():
    print("Main Driver")

    style_name = 'sofa-1.jpg'

    style_path = os.path.join(D.STYLE_DIR.get(),style_name)
    _,mask_filepath = segment_points(style_path,device=D.DEVICE())

    style = utils.image_to_tensor(utils.load_image(style_path))

    mask = utils.image_to_tensor(utils.load_image(mask_filepath))

    cropped = style * mask 

    final = utils.tensor_to_image(cropped)
    final_name = style_name+'_cropped.png'
    save_path = os.path.join(D.STYLE_DIR.get(),final_name)
    final.save(save_path)





    # args = parse_arguments()
    
    # Load style furniture image
    # style = utils.image_to_tensor(utils.load_image(args.style)).detach()
    
    # mesh = utils.load_mesh(args.mesh)
    
    # # Create model for 3D texture transfer
    # render_model = MeshRendererModel(mesh,style)
    
    # texture_transfer_gatys(render_model,style)



if __name__ == "__main__":
    main()
    






   
