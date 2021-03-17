from dextr import segment

import helpers.utils as utils
from defaults import DEFAULTS as D

import pathlib as p
import os

import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(
        description='UV Map Retrieval. Uses Blender API to obtain the UV maps of meshes')
    parser.add_argument('--image',type=str,help='Path of image to mask')
    parser.add_argument('--output',type=str,help='Path to save output masked image')
    
    return parser.parse_args()

def mask(mask_path, image_path,output_path):
    mask = utils.load_image(mask_path,mode="L")
    style = utils.load_image(image_path,mode="RGBA")
    output = style.copy()
    output.putalpha(mask)
    output.save(output_path,'PNG')
    return output_path

def segment_(image_path,device= D.DEVICE()):
    mask_path = segment.segment_points(image_path=image_path, device=device)
    return mask_path


if __name__=='__main__':
    device = D.DEVICE()
    args = parse_arguments()
    image_path = args.image
    mask_path = segment.segment_points(image_path=image_path, device=device)
    output_path = args.output
    mask(mask_path,image_path,output_path)
    os.remove(mask_path)

    # img = utils.load_image(image_path)
    # tensor = utils.image_to_tensor(img)
    # mask = tensor[:,3,...].unsqueeze(0)
    # mask_img = utils.tensor_to_image(mask,denorm=False)
    # mask_img.save('uv_map_backseat_chair-2_masked_mask.png')

