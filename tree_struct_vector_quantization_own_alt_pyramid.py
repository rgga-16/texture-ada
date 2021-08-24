from numpy.core.numeric import False_
import torch
from torchvision.transforms import GaussianBlur
import torch.nn.functional as F
from torchvision.transforms.functional import scale
from torchvision.transforms.transforms import ToPILImage
import torchvision.transforms as T

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

def get_neighborhood(curr_row,curr_col,image,n_size,exclude_curr_pixel=True):
    im = image.clone().detach()
    im_c,im_h,im_w = image.shape

    if type(n_size) is tuple:
        n_h, n_w = n_size
    else: 
        n_h = n_w =  n_size

    half_h = n_h//2
    half_w = n_w//2

    top_idx = curr_row-half_h
    bot_idx = curr_row+half_h
    left_idx = curr_col-half_w
    right_idx= curr_col+half_w

    top_pad = bot_pad=left_pad=right_pad=0
    
    if top_idx < 0: 
        top_pad = abs(top_idx)
    if bot_idx >= im_h:
        bot_pad = abs(im_h-1-bot_idx)
    if left_idx < 0:
        left_pad = abs(left_idx)
    if right_idx >= im_w:
        right_pad = abs(im_w-1-right_idx)

    top_idx = np.clip(curr_row-half_h,0,im_h-1)
    bot_idx = np.clip(curr_row+half_h,0,im_h-1)
    left_idx = np.clip(curr_col-half_w,0,im_w-1)
    right_idx= np.clip(curr_col+half_w,0,im_w-1)

    if exclude_curr_pixel:
        # curr_idx = curr_row*n_cols + curr_col
        im[:,curr_row,curr_col]=0
        # neighborhood = torch.cat((neighborhood[:,0:curr_idx],neighborhood[:,curr_idx+1:]),dim=-1)

    neighborhood = im[:,top_idx:bot_idx+1,left_idx:right_idx+1]
    neighborhood = F.pad(neighborhood,(left_pad,right_pad,top_pad,bot_pad),value=0)
    _,n_rows,n_cols = neighborhood.shape

    # neighborhood = neighborhood.reshape(neighborhood.shape[0],-1)

    neighborhood[neighborhood==-float('inf')]=0

    # if neighborhood.shape[1] > n_h:
    #     neighborhood = neighborhood[:,:n_h,:]
    
    # if neighborhood.shape[2] > n_w:
    #     neighborhood = neighborhood[:,:,:n_w]

    
    return neighborhood

def get_neighborhood_pyramid(curr_row,curr_col,pyramid,level,n_size,n_parent_size,exclude_curr_pixel):
    N = get_neighborhood(curr_row,curr_col,pyramid[level],n_size,exclude_curr_pixel)
    if level > 0:
        parent_row = curr_row//2
        parent_col = curr_col//2
        parent_N = get_neighborhood(parent_row,parent_col,pyramid[level-1],n_parent_size,exclude_curr_pixel)

        N = torch.stack((N,parent_N))

 
    # curr_N = curr_N.reshape(curr_N.shape[0],-1)
    # prev_N = prev_N.reshape(prev_N.shape[0],-1)
    
    # N = torch.cat((curr_N,prev_N),dim=-1)
    # N_summed = torch.sum(N,dim=-1)
    return N

def get_neighborhood_pyramids(pyramid,level,n_size,n_parent_size,exclude_curr_pixel):
    kD_pixels=[]
    neighborhood_pyrs=[]
    image = pyramid[level]
    _,h,w = image.shape 

    for r in range(h):
        for c in range(w):
            kD_pixels.append(pyramid[level][:,r,c])
            N = get_neighborhood_pyramid(r,c,pyramid,level,n_size,n_parent_size,exclude_curr_pixel)
            neighborhood_pyrs.append(N)
            # neighborhood_pyrs.append(torch.cat((curr_N.reshape(curr_N.shape[0],-1),prev_N.reshape(prev_N.shape[0],-1)),dim=-1))
            
    stacked = torch.stack(neighborhood_pyrs)
    return torch.stack(neighborhood_pyrs),torch.stack(kD_pixels)

def find_best_match(G_a,G_s,L,x_s,y_s,n_size,n_parent_size,N_a_list,Cs):
    N_s = build_neighborhood(G_s,L,x_s,y_s,n_size,exclude_center_pixel=True)
    if(L-1 >= 0):
        N_s_parent = build_neighborhood(G_s,L-1,x_s//2,y_s//2,n_parent_size,exclude_center_pixel=False)
        N_s = torch.cat((N_s,N_s_parent))
    N_s = N_s.sum(0,keepdim=True)
    
    dists = F.pairwise_distance(N_a_list,N_s,p=2)
    best_idx = torch.argmin(dists)
    C = Cs[best_idx]

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
    number_of_images = len(images)
    for i in range(number_of_images):
        a=fig.add_subplot(1,number_of_images,i+1)
        image = images[i]
        imshow(image.permute(1,2,0).detach().cpu().numpy())
        axis('off')
    plt.show()

def tvsq_new(input_path,output_path,n_size,n_levels):

    # if type(n_size) is tuple:
    #     n_h,n_w = n_size 
    #     n_parent_size = (math.ceil(n_h/2), math.ceil(n_w/2))
    # else:
    #     n_parent_size = math.ceil(n_size/2)
    # n_parent_size = n_size

    I_a = image_utils.image_to_tensor(image_utils.load_image(input_path,mode='RGB'),image_size=D.IMSIZE.get()//2,normalize=False)   

    I_s = torch.empty(3,D.IMSIZE.get(),D.IMSIZE.get(),device=D.DEVICE()).fill_(-float('inf'))

    G_a = build_gaussian_pyramid(I_a,n_levels=n_levels)
    G_s = build_gaussian_pyramid(I_s,n_levels=n_levels)

    # showImagesHorizontally(G_a)

    n_sizes = [3,5,7,9]
    for L in range(n_levels):
        print(f'Level {L+1} of {n_levels}')
        _,h_a,w_a = G_a[L].shape
        N_a_list = [] #Get N_a_list after each level is introduced.
        C_candids = []
        n_size = n_sizes[L]

        for y_a in range(h_a):
            for x_a in range(w_a):
                N_a = build_neighborhood(G_a,L,x_a,y_a,n_size,exclude_center_pixel=False)
                if(L-1 >= 0):
                    n_parent_size = n_size
                    N_a_parent = build_neighborhood(G_a,L-1,x_a//2,y_a//2,n_parent_size,exclude_center_pixel=False)
                    N_a = torch.cat((N_a,N_a_parent))
                N_a = N_a.sum(0)

                N_a_list.append(N_a)
                C_candids.append(G_a[L][:,y_a,x_a])
        
        N_a_list = torch.stack(N_a_list)

        _,h_s,w_s = G_s[L].shape
        for y_s in range(h_s):
            print(f'Row {y_s+1} of {h_s}')
            for x_s in range(w_s):
                C = find_best_match(G_a,G_s,L,x_s,y_s,n_size,n_parent_size,N_a_list,C_candids)
                G_s[L][:,y_s,x_s]=C 
    showImagesHorizontally(G_a)
    showImagesHorizontally(G_s)
    
    final_I_s = G_s[-1]
    out_im = image_utils.tensor_to_image(final_I_s,denorm=False)
    out_im.save(output_path)
    return final_I_s

def tvsq(input_path,output_path,n_size,n_levels):
    if type(n_size) is tuple:
        n_h,n_w = n_size 
        parent_size = (math.ceil(n_h/2), math.ceil(n_w/2))
    else:
        parent_size = math.ceil(n_size/2)
    
    I_a = image_utils.image_to_tensor(image_utils.load_image(input_path,mode='RGB'),image_size=D.IMSIZE.get(),normalize=False)
    I_s = torch.empty(3,D.IMSIZE.get(),D.IMSIZE.get(),device=D.DEVICE()).fill_(-float('inf'))

    random_crop = T.RandomCrop(n_size)
    cropped = random_crop(I_a)
    image_utils.tensor_to_image(cropped,image_size=D.IMSIZE.get(),denorm=False).save('cropped.png')
    
    G_a = build_gaussian_pyramid(I_a,n_levels=n_levels)
    G_s = build_gaussian_pyramid(I_s,n_levels=n_levels)
    n_sizes = [9,17,33,65]
    for L in range(n_levels):
        n_size=n_sizes[L]
        parent_size=n_size
        neighborhoods_pyr,kD_pixels = get_neighborhood_pyramids(G_a,L,n_size,parent_size,False)
        _,o_h,o_w = G_s[L].shape 
        for o_r in range(o_h):
            print(f'Rows {o_r+1} of {o_h}')
            for o_c in range(o_w):
                # curr_N,prev_N = get_neighborhood_pyramid(o_r,o_c,G_s,L,n_size,parent_size,exclude_curr_pixel=True)
                N_o = get_neighborhood_pyramid(o_r,o_c,G_s,L,n_size,parent_size,exclude_curr_pixel=True).unsqueeze(0)
                # N_o = torch.cat((curr_N.reshape(curr_N.shape[0],-1),prev_N.reshape(prev_N.shape[0],-1)),dim=-1)
                dists = F.pairwise_distance(N_o,neighborhoods_pyr,p=2,keepdim=True).squeeze()
                if L==0:
                    dists = torch.sum(dists,dim=(-1,-2))
                else: 
                    dists = torch.sum(dists,dim=(-1,-2,-3))
                # dists = F.pairwise_distance(N_o.reshape(1,-1),neighborhoods_pyr.reshape(neighborhoods_pyr.shape[0],-1),p=2,keepdim=True)
                best_idx = torch.argmin(dists)
                best_match = kD_pixels[best_idx]

                # pixel = get_central_pixel(best_match,n_size)
                G_s[L][:,o_r,o_c]=best_match

                # Insert best match into output texture
    showImagesHorizontally(G_s)
    print()

    final_output = G_s[-1]

    out_im = image_utils.tensor_to_image(final_output,denorm=False)
    out_im.save(output_path)
    return final_output


def main():

    start = timer()
    n_size=65
    n_lvls=4
    input_path = './inputs/style_images/fdf_textures/12.png'
    # output = tvsq_new(input_path,f'{os.path.basename(input_path[:-4])}_{n_size}_alt_pyramid_{n_lvls}lvls_new.png',None,n_lvls)
    output = tvsq(input_path,f'{os.path.basename(input_path[:-4])}_{n_size}_alt_pyramid_{n_lvls}lvls.png',n_size=n_size,n_levels=n_lvls)
    end=timer()
    print(f'Time elapsed: {end-start:.2f}')
    return


if __name__ == "__main__":   
    main()