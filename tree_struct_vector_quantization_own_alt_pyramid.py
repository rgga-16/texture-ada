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

def get_n_h_and_n_w(n_size):
    if type(n_size) is tuple:
        n_h, n_w = n_size
    else: 
        n_h = n_w =  n_size
    
    return n_h,n_w

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
        # parent_row = int(round((curr_row/curr_h) * parent_h))
        # parent_col = int(round((curr_col/curr_w) * parent_w))
        parent_row = curr_row//2
        parent_col = curr_col//2
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

def find_best_match(G_a,G_s,L,x_s,y_s,n_size,n_parent_size):
    N_a_best = None; C=None 
    N_s_child = build_neighborhood(G_s,L,x_s,y_s,n_size,exclude_center_pixel=True)
    if(L-1 >= 0):
        N_s_parent = build_neighborhood(G_s,L-1,x_s//2,y_s//2,n_parent_size,exclude_center_pixel=False)
        N_s_child = torch.cat((N_s_child,N_s_parent))
    N_s_child = N_s_child.sum(0,keepdim=True)
    
    _,h_a,w_a = G_a[L].shape
    for y_a in range(h_a):
        for x_a in range(w_a):
            N_a_child = build_neighborhood(G_a,L,x_a,y_a,n_size,exclude_center_pixel=False)
            if(L-1 >= 0):
                N_a_parent = build_neighborhood(G_a,L-1,x_a//2,y_a//2,n_parent_size,exclude_center_pixel=False)
                N_a_child = torch.cat((N_a_child,N_a_parent))
            N_a_child = N_a_child.sum(0,keepdim=True)
            
            diff = F.mse_loss(N_s_child,N_a_child,reduction='sum')
            if(N_a_best is None or (diff < F.mse_loss(N_s_child,N_a_best,reduction='sum'))):
                N_a_best = N_a_child 
                C = G_a[L][:,y_a,x_a]
    return C

def build_neighborhood(G,L,x_g,y_g,n_size,exclude_center_pixel=True):
    N=[]
    _,im_h,im_w = G[L].shape
    
    n_h,n_w = get_n_h_and_n_w(n_size)
    half_h = n_h//2
    half_w = n_w//2

    start_x = x_g - half_w
    start_y = y_g - half_h 
    end_x = x_g+half_w 
    end_y = y_g+half_h 

    if start_x < 0: 
        start_x=0
    if end_x > im_h:
        end_x=im_w
    if start_y < 0:
        start_y=0
    if end_y > im_h:
        end_y=im_h

    for y in range(start_y,end_y):
        for x in range(start_x,end_x):
            if G[L][0,y,x]<0:
                continue 
            elif x==x_g and y==y_g:
                if exclude_center_pixel:
                    continue 
                else: N.append(G[L][:,y,x])
            else: 
                N.append(G[L][:,y,x])

    # NEIGHBORHOOD OF OUTPUT PIXEL SHOULD ONLY HAVE PREVIOUSLY DETERMINED PIXEL VALUES. undiscovered values
    # should be removed.
    #find a way to return neighborhood as shape (N,3)
    # Use conditions if len(N) is 0,1 or more than 1
    if(len(N)==0):
        yes=torch.zeros(1,3,device=D.DEVICE())
    else:
        yes= torch.stack(N)

    return yes


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

from matplotlib.pyplot import figure, imshow, axis
from matplotlib.image import imread
import matplotlib.pyplot as plt
def showImagesHorizontally(images):
    fig = figure()
    number_of_files = len(images)
    for i in range(number_of_files):
        a=fig.add_subplot(1,number_of_files,i+1)
        image = images[i]
        imshow(image.permute(1,2,0).detach().cpu().numpy())
        axis('off')
    plt.show()

def tvsq_new(input_path,output_path,n_size,n_levels):

    if type(n_size) is tuple:
        n_h,n_w = n_size 
        n_parent_size = (math.ceil(n_h/2), math.ceil(n_w/2))
    else:
        n_parent_size = math.ceil(n_size/2)

    I_a = image_utils.image_to_tensor(image_utils.load_image(input_path,mode='RGB'),image_size=D.IMSIZE.get(),normalize=False)   
    I_s = torch.neg(torch.rand(3,D.IMSIZE.get(),D.IMSIZE.get(),device=D.DEVICE())).detach()

    G_a = build_gaussian_pyramid(I_a,n_levels=n_levels)
    G_s = build_gaussian_pyramid(I_s,n_levels=n_levels)

    # showImagesHorizontally(G_a)

    for L in range(n_levels):
        print(f'Level {L}')
        _,h_s,w_s = G_s[L].shape
        for y_s in range(h_s):
            print(f'Row {y_s+1} of {h_s+1}')
            for x_s in range(w_s):
                C = find_best_match(G_a,G_s,L,x_s,y_s,n_size,n_parent_size)
                G_s[L][:,y_s,x_s]=C 
    final_I_s = G_s[-1]
    return final_I_s

def tvsq(input_path,n_size,n_levels):
    
    if type(n_size) is tuple:
        n_h,n_w = n_size 
        parent_size = (math.ceil(n_h/2), math.ceil(n_w/2))
    else:
        parent_size = math.ceil(n_size/2)
    
    input_texture = image_utils.image_to_tensor(image_utils.load_image(input_path,mode='RGB'),image_size=D.IMSIZE.get(),normalize=False)
    output_texture = torch.neg(torch.rand(3,D.IMSIZE.get(),D.IMSIZE.get(),device=D.DEVICE())).detach()

    input_texture_pyramid = build_gaussian_pyramid(input_texture,n_levels=n_levels)
    output_texture_pyramid = build_gaussian_pyramid(output_texture,n_levels=n_levels)

    build_neighborhood(input_texture_pyramid,-1,100,100,n_size,exclude_center_pixel=False)
   
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
    n_size=5
    n_lvls=4
    input_path = './inputs/style_images/fdf_textures/23.png'
    output_path=None
    tvsq_new(input_path,output_path,n_size,n_lvls)
    # output = tvsq('./inputs/style_images/fdf_textures/23.png',None,n_size=n_size,n_levels=n_lvls)
    # out_im = image_utils.tensor_to_image(output,denorm=False)
    # out_im.save(f'output_{n_size}_alt_pyramid_{n_lvls}lvls.png')
    end=timer()
    print(f'Time elapsed: {end-start:.2f}')
    return


if __name__ == "__main__":   
    main()