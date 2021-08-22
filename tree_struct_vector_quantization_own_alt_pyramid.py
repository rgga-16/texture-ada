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
        pyramid.insert(0,im.squeeze())

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
    
    if neighborhood.shape[1] > n_h:
        neighborhood = neighborhood[:,:n_h,:]
    
    if neighborhood.shape[2] > n_w:
        neighborhood = neighborhood[:,:,:n_w]
    
    return neighborhood

def get_neighborhood_pyramid(curr_row,curr_col,pyramid,level,n_size,n_parent_size):
    curr_N = get_neighborhood(curr_row,curr_col,pyramid[level],n_size)
    if level > 0:
        _,curr_h,curr_w = pyramid[level].shape
        _,parent_h,parent_w = pyramid[level-1].shape
        parent_row = int(round((curr_row/curr_h) * parent_h))
        parent_col = int(round((curr_col/curr_w) * parent_w))
        prev_N = get_neighborhood(parent_row,parent_col,pyramid[level-1],n_parent_size)
    else:
        prev_N = torch.zeros(3,n_parent_size,n_parent_size,device=D.DEVICE())

    
    # curr_N = curr_N.reshape(curr_N.shape[0],-1)
    # prev_N = prev_N.reshape(prev_N.shape[0],-1)
    
    # N = torch.cat((curr_N,prev_N),dim=-1)
    return curr_N,prev_N

def get_neighborhood_pyramids(pyramid,level,n_size,n_parent_size):
    neighborhoods=[]
    neighborhood_pyrs=[]
    image = pyramid[level]
    _,h,w = image.shape 

    for r in range(h):
        for c in range(w):
            curr_N, prev_N = get_neighborhood_pyramid(r,c,pyramid,level,n_size,n_parent_size)
            neighborhood_pyrs.append(torch.cat((curr_N.reshape(curr_N.shape[0],-1),prev_N.reshape(prev_N.shape[0],-1)),dim=-1))
            neighborhoods.append(curr_N)
    
    return torch.stack(neighborhood_pyrs),torch.stack(neighborhoods)

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
    
    input_texture = image_utils.image_to_tensor(image_utils.load_image(input_path,mode='RGB'),image_size=D.IMSIZE.get(),normalize=False)
    output_texture = torch.zeros(3,D.IMSIZE.get(),D.IMSIZE.get(),device=D.DEVICE()).detach()

    input_texture_pyramid = build_gaussian_pyramid(input_texture,n_levels=n_levels)
    output_texture_pyramid = build_gaussian_pyramid(output_texture,n_levels=n_levels)

    if type(n_size) is tuple:
        n_h, n_w = n_size
    else: 
        n_h = n_w =  n_size
   
    for l in range(n_levels):
        neighborhoods_pyr,neighborhoods = get_neighborhood_pyramids(input_texture_pyramid,l,n_size,parent_size)
        _,o_h,o_w = output_texture_pyramid[l].shape 
        for o_r in range(o_h):
            print(f'Rows {o_r} of {o_h-1}')
            for o_c in range(o_w):
                curr_N,prev_N = get_neighborhood_pyramid(o_r,o_c,output_texture_pyramid,l,n_size,parent_size)

                N_o = torch.cat((curr_N.reshape(curr_N.shape[0],-1),prev_N.reshape(prev_N.shape[0],-1)),dim=-1)

                dists = F.pairwise_distance(N_o.reshape(1,-1),neighborhoods_pyr.reshape(neighborhoods_pyr.shape[0],-1),p=2,keepdim=True)
                best_idx = torch.argmin(dists)
                best_match = neighborhoods[best_idx]

                pixel = get_central_pixel(best_match,n_size)
                output_texture_pyramid[l][:,o_r,o_c]=pixel

                # Insert best match into output texture
        output = image_utils.tensor_to_image(output_texture_pyramid[l],denorm=False)
        output.show()
        print()


    final_output = output_texture_pyramid[-1]


    return final_output


def main():

    start = timer()
    n_size=48
    n_lvls=1
    output = tvsq('./inputs/style_images/fdf_textures/23.png',None,n_size=n_size,n_levels=n_lvls)
    out_im = image_utils.tensor_to_image(output,denorm=False)
    out_im.save(f'output_{n_size}_alt_pyramid_{n_lvls}lvls.png')
    end=timer()
    print(f'Time elapsed: {end-start:.2f}')
    return


if __name__ == "__main__":   
    main()