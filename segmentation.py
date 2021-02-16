from dextr import segment

import args as args
import helpers.utils as utils
from defaults import DEFAULTS as D

import pathlib as p
import os

def mask(mask_path, style_path,output_path):

    mask = utils.load_image(mask_path,mode="L")

    style = utils.load_image(style_path,mode="RGBA")

    output = style.copy()

    output.putalpha(mask)

    output.save(output_path,'PNG')

    return output_path


if __name__=='__main__':
    device = D.DEVICE()
    image_path = './inputs/style_images/chair-6.jpg'
    mask_path = segment.segment_points(image_path=image_path, device=device)
    mask_path = 'uv_map.png'
    image_path = '_final.png'
    output_path = 'texture_map.png'
    mask(mask_path,image_path,output_path)

