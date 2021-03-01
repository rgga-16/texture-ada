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
    image_path = './uv_map_backseat_chair-2_masked.png'
    # mask_path = segment.segment_points(image_path=image_path, device=device)
    # mask_path = 'uv_map.png'
    # image_path = '_final.png'
    # output_path = 'texture_map.png'
    # mask(mask_path,image_path,output_path)

    img = utils.load_image(image_path)
    tensor = utils.image_to_tensor(img)
    mask = tensor[:,3,...].unsqueeze(0)
    mask_img = utils.tensor_to_image(mask,denorm=False)
    mask_img.save('uv_map_backseat_chair-2_masked_mask.png')

