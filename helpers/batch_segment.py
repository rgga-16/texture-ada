import helpers.utils as utils
from defaults import DEFAULTS as D

import pathlib as p
import os
from re import search

import segmentation 

if __name__=='__main__':

    dir = './inputs/style_images'
    output_dir = './inputs/style_images/masked'

    for file in os.listdir(dir):
        if search('cobonpue',file):
            output_path = os.path.join(output_dir,'{}_masked.png'.format(file[:-4]))
            image_path = os.path.join(dir,file)
            mask_path = segmentation.segment_(image_path)
            segmentation.mask(mask_path,image_path,output_path)
    