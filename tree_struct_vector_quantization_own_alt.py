import torch
from torchvision.transforms import GaussianBlur
import torch.nn.functional as F
from torchvision.transforms.functional import scale
from torchvision.transforms.transforms import ToPILImage

from defaults import DEFAULTS as D

import helpers.seeder as seeder
from helpers import image_utils
import args as args_

import random, numpy as np, os
from timeit import default_timer as timer

def build_gaussian_pyramid(image,n_levels):

    gaussian_blur = GaussianBlur(kernel_size=(5,5))
    pyramid = [image]
    im = image.clone().detach().unsqueeze(0)
    for l in range(0,n_levels-1):
        _,_,h,w = im.shape
        blurred_im = gaussian_blur(im)
        downsampled_im = F.interpolate(blurred_im,scale_factor=0.5)
        im = downsampled_im
        pyramid.append(im.squeeze())

    return pyramid

def get_neighborhood(curr_row,curr_col,image,n_size):
    _,im_h,im_w = image.shape

    if type(n_size) is tuple:
        n_h, n_w = n_size
    else: 
        n_h = n_w =  n_size

    half_h = n_h//2
    half_w = n_w//2

    # shifted_row = curr_row+half_h
    # shifted_col = curr_col+half_w

    top_idx = curr_row-half_h
    bot_idx = curr_row+half_h
    left_idx = curr_col-half_w
    right_idx= curr_col+half_w
    
    top_pad = bot_pad=left_pad=right_pad=0
    
    if top_idx < 0: 
        top_pad = abs(top_idx)
        top_idx=0
    if bot_idx >= im_h:
        bot_pad = abs(im_h-1-bot_idx)
        bot_idx=im_h-1
    if left_idx < 0:
        left_pad = abs(left_idx)
        left_idx = 0
    if right_idx >= im_w:
        right_pad = abs(im_w-1-right_idx)
        right_idx = im_w-1    

    neighborhood = image[:,top_idx:bot_idx+1,left_idx:right_idx+1]
    neighborhood = F.pad(neighborhood,(left_pad,right_pad,top_pad,bot_pad),value=0)
    return neighborhood

def get_neighborhoods(image,n_size):
    neighborhoods=[]
    _,h,w = image.shape 

    if type(n_size) is tuple:
        n_h, n_w = n_size
    else: 
        n_h = n_w =  n_size
        
    for r in range(h):
        for c in range(w):
            neighborhoods.append(get_neighborhood(r,c,image,n_size))
    
    return neighborhoods

def get_central_pixel(neighborhood, n_size):

    if type(n_size) is tuple:
        n_h, n_w = n_size
    else: 
        n_h = n_w =  n_size
    
    half_h = n_h//2; half_w = n_w//2

    return neighborhood[:,half_h,half_w]

def tvsq(input_path, output_path,n_size,n_levels):
    input_texture = image_utils.image_to_tensor(image_utils.load_image(input_path,mode='RGB'),normalize=False)
    output_texture = torch.rand(3,D.IMSIZE.get(),D.IMSIZE.get(),device=D.DEVICE()).detach()

    # input_texture = torch.rand(1,9,9,device=D.DEVICE())
    # output_texture = torch.rand(1,9,9,device=D.DEVICE())
    
    neighborhoods = get_neighborhoods(input_texture,n_size)
    n_neighborhoods = len(neighborhoods)
    neighborhoods = torch.stack(neighborhoods)

    if type(n_size) is tuple:
        n_h, n_w = n_size
    else: 
        n_h = n_w =  n_size

    _,o_h,o_w = output_texture.shape 
    for o_r in range(o_h):
        print(f'Rows {o_r} of {o_h-1}')
        for o_c in range(o_w):
            N_o = get_neighborhood(o_r,o_c,output_texture,n_size)
            dists = F.pairwise_distance(N_o.reshape(1,-1),neighborhoods.reshape(neighborhoods.shape[0],-1))
            best_idx = torch.argmin(dists)
            best_match = neighborhoods[best_idx]

            pixel = get_central_pixel(best_match,n_size)
            output_texture[:,o_r,o_c]=pixel

            # half_h = n_h//2
            # half_w = n_w//2

            # shifted_row = o_r+half_h
            # shifted_col = o_c+half_w

            # top_idx = shifted_row-half_h
            # bot_idx = shifted_row+half_h
            # left_idx = shifted_col-half_w
            # right_idx= shifted_col+half_w
            # output_texture[:,top_idx:bot_idx+1,left_idx:right_idx+1]=best_match
            # Insert best match into output texture

    
    # input_texture_pyramid = build_gaussian_pyramid(input_texture,n_levels=n_levels)


    return output_texture


def main():

    start = timer()
    n_size=20
    output = tvsq('./inputs/style_images/fdf_textures/12.png',None,n_size=n_size,n_levels=4)
    out_im = image_utils.tensor_to_image(output,denorm=False)
    out_im.save(f'output_{n_size}.png')
    end=timer()
    print(f'Time elapsed: {end-start:.2f}')
    return


if __name__ == "__main__":   
    main()