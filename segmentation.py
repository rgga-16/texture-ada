from external_libs.dextr import segment

import helpers.image_utils as utils
from defaults import DEFAULTS as D

import pathlib as p
import os

import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Segmentor')
    parser.add_argument('--image_dir',type=str,help='Path to dir of images to mask')
    
    return parser.parse_args()

def mask(mask_path, image_path):
    mask = utils.load_image(mask_path,mode="L")
    image = utils.load_image(image_path,mode="RGBA")
    masked = image.copy()
    masked.putalpha(mask)
    return masked,mask
def segment_(image_path,device= D.DEVICE()):
    mask_path = segment.segment_points(image_path=image_path, device=device)
    return mask_path


if __name__=='__main__':
    device = D.DEVICE()
    args = parse_arguments()

    output_folder = os.path.join(args.image_dir, 'masked')
    try:
        os.mkdir(output_folder)
    except FileExistsError:
        pass
    
    mask_folder = os.path.join(args.image_dir, 'masks')
    try:
        os.mkdir(mask_folder)
    except FileExistsError:
        pass

    for f in os.listdir(args.image_dir):
        if(not os.path.isfile(os.path.join(args.image_dir,f))):
            continue
        image_path = os.path.join(args.image_dir,f)
        temp_mask_path = segment.segment_points(image_path=image_path, device=device)
        masked_img, mask_ = mask(temp_mask_path,image_path)
        masked_img.save(os.path.join(output_folder,f'{f[:-4]}.png'))
        mask_.save(os.path.join(mask_folder,f'{f[:-4]}.png'))
        os.remove(temp_mask_path)


