import torch
from torchvision.transforms import GaussianBlur
import torch.nn.functional as F
from torchvision.transforms.functional import scale
from torchvision.transforms.transforms import ToPILImage

from defaults import DEFAULTS as D

import helpers.seeder as seeder
from helpers import image_utils
import args as args_

import random, numpy as np, os, math
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

def get_neighborhood_pyramid(curr_row,curr_col,pyramid,level,n_size,n_parent_size):
    curr_N = get_neighborhood(curr_row,curr_col,pyramid[level],n_size)
    if level > 0:
        prev_N = get_neighborhood(curr_row,curr_col,pyramid[level-1],n_parent_size)
        print()
    else:
        prev_N = torch.zeros(3,n_parent_size,n_parent_size,device=D.DEVICE())
    
    curr_N = curr_N.reshape(curr_N.shape[0],-1)
    prev_N = prev_N.reshape(prev_N.shape[0],-1)
    
    N = torch.cat((curr_N,prev_N),dim=-1)
    return N

def get_neighborhood_pyramids(pyramid,level,n_size,n_parent_size):
    neighborhoods=[]
    image = pyramid[level]
    _,h,w = image.shape 

    for r in range(h):
        for c in range(w):
            neighborhoods.append(get_neighborhood_pyramid(r,c,pyramid,level,n_size,n_parent_size))

    return neighborhoods

def get_neighborhoods(image,n_size):
    neighborhoods=[]
    _,h,w = image.shape 
        
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
    
    if type(n_size) is tuple:
        n_h,n_w = n_size 
        parent_size = (math.ceil(n_h/2), math.ceil(n_w/2))
    else:
        parent_size = math.ceil(n_size/2)
    
    input_texture = image_utils.image_to_tensor(image_utils.load_image(input_path,mode='RGB'),normalize=False)
    output_texture = torch.rand(3,D.IMSIZE.get(),D.IMSIZE.get(),device=D.DEVICE()).detach()

    input_texture_pyramid = build_gaussian_pyramid(input_texture,n_levels=n_levels)
    output_texture_pyramid = build_gaussian_pyramid(output_texture,n_levels=n_levels)
    
    # input_texture = torch.rand(1,9,9,device=D.DEVICE())
    # output_texture = torch.rand(1,9,9,device=D.DEVICE())
    
    # neighborhoods = get_neighborhoods(input_texture,n_size)
    # neighborhoods = torch.stack(neighborhoods)

    if type(n_size) is tuple:
        n_h, n_w = n_size
    else: 
        n_h = n_w =  n_size

    _,o_h,o_w = output_texture.shape 
    for l in range(n_levels):
        neighborhoods = get_neighborhood_pyramids(input_texture_pyramid,l,n_size,parent_size)
        neighborhoods = torch.stack(neighborhoods)
        for o_r in range(o_h):
            print(f'Rows {o_r} of {o_h-1}')
            for o_c in range(o_w):
                N_o = get_neighborhood_pyramid(o_r,o_c,output_texture_pyramid,l,n_size,parent_size)
                dists = F.pairwise_distance(N_o.reshape(1,-1),neighborhoods.reshape(neighborhoods.shape[0],-1),p=2,keepdim=True)
                best_idx = torch.argmin(dists)
                best_match = neighborhoods[best_idx]

                pixel = get_central_pixel(best_match,n_size)
                output_texture[:,o_r,o_c]=pixel

                # Insert best match into output texture

    
    # input_texture_pyramid = build_gaussian_pyramid(input_texture,n_levels=n_levels)


    return output_texture


def main():

    start = timer()
    n_size=5
    output = tvsq('./inputs/style_images/fdf_textures/12.png',None,n_size=n_size,n_levels=4)
    out_im = image_utils.tensor_to_image(output,denorm=False)
    out_im.save(f'output_{n_size}.png')
    end=timer()
    print(f'Time elapsed: {end-start:.2f}')
    return


if __name__ == "__main__":   
    main()