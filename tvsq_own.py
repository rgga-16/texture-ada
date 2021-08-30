from numpy.core.numeric import False_
import torch
from torchvision.transforms import GaussianBlur
import torch.nn.functional as F
from torchvision.transforms.functional import scale
from torchvision.transforms.transforms import ToPILImage
import torchvision.transforms as T

from defaults import DEFAULTS as D
from ops import ops
import args as args_

import helpers.seeder as seeder
from helpers import image_utils
import args as args_

import random, numpy as np, os, math
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure, imshow, axis
from matplotlib.image import imread
from timeit import default_timer as timer

def log_dists(path,dists):
    with open(path,'w') as f:
        f.writelines('\n'.join(dists))
    return

def get_wassdist(ref_texture,output_texture):

    style_means,style_covs = ops.get_means_and_covs(ref_texture)
    output_means,output_covs = ops.get_means_and_covs(output_texture)

    wass_dist = 0
    for s in D.STYLE_LAYERS.get().values():
        wdist = ops.gaussian_wasserstein_distance(style_means[s],style_covs[s],output_means[s],output_covs[s]).real
        wass_dist += D.SL_WEIGHTS.get()[s] * wdist 
    wasserstein_distance = torch.mean(wass_dist)
    
    return wasserstein_distance.item()

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
    _,im_h,im_w = im.shape

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
        im[:,curr_row,curr_col]=0
        # im[:,curr_row,curr_col+1:]=0
        # im[:,curr_row+1:,:]=0
        
    neighborhood = im[:,top_idx:bot_idx+1,left_idx:right_idx+1]
    neighborhood = F.pad(neighborhood,(left_pad,right_pad,top_pad,bot_pad),value=0)

    return neighborhood

def get_neighborhood_pyramid(curr_row,curr_col,pyramid,level,n_size,n_parent_size,exclude_curr_pixel):
    N = get_neighborhood(curr_row,curr_col,pyramid[level],n_size,exclude_curr_pixel)
    if level > 0:
        parent_row = curr_row//2
        parent_col = curr_col//2
        parent_N = get_neighborhood(parent_row,parent_col,pyramid[level-1],n_parent_size,False)

        if parent_N.shape[1] < N.shape[1]:
            diff = abs(N.shape[1] - parent_N.shape[1])
            parent_N = F.pad(parent_N,(int(diff/2),int(diff/2),int(diff/2),int(diff/2)),value=0)

        N = torch.stack((N,parent_N))



    

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

    return torch.stack(neighborhood_pyrs),torch.stack(kD_pixels)


def showImagesHorizontally(images):
    fig = figure()
    number_of_images = len(images)
    for i in range(number_of_images):
        a=fig.add_subplot(1,number_of_images,i+1)
        image = images[i]
        imshow(image.permute(1,2,0).detach().cpu().numpy())
        axis('off')
    plt.show()

def tvsq(input_path,output_path,n_size,n_levels,in_size=D.IMSIZE.get(),out_size=D.IMSIZE.get()):
    if type(n_size) is tuple:
        n_h,n_w = n_size 
        parent_size = (math.ceil(n_h/2), math.ceil(n_w/2))
    else:
        parent_size = math.ceil(n_size/2)
    
    I_a = image_utils.image_to_tensor(image_utils.load_image(input_path,mode='RGB'),image_size=in_size,normalize=False)
    I_s = torch.rand(3,out_size,out_size,device=D.DEVICE()).detach()
    # I_s = torch.empty(3,out_size,out_size,device=D.DEVICE()).fill_(-float('inf'))

    random_crop = T.RandomCrop(n_size)
    cropped = random_crop(I_a)
    image_utils.tensor_to_image(cropped,image_size=out_size,denorm=False).save('cropped.png')
    
    G_a = build_gaussian_pyramid(I_a,n_levels=n_levels)
    G_s = build_gaussian_pyramid(I_s,n_levels=n_levels)
    # parent_size=n_size
    for L in range(n_levels):
        neighborhoods_pyr,kD_pixels = get_neighborhood_pyramids(G_a,L,n_size,parent_size,False)
        _,o_h,o_w = G_s[L].shape 
        for o_r in range(o_h):
            print(f'Rows {o_r+1} of {o_h}')
            for o_c in range(o_w):
                N_o = get_neighborhood_pyramid(o_r,o_c,G_s,L,n_size,parent_size,exclude_curr_pixel=True).unsqueeze(0)
                dists = F.pairwise_distance(N_o,neighborhoods_pyr,p=2,keepdim=True).squeeze()
                if L==0:
                    dists = torch.sum(dists,dim=(-1,-2))
                else: 
                    dists = torch.sum(dists,dim=(-1,-2,-3))
               
                best_idx = torch.argmin(dists)
                best_match = kD_pixels[best_idx]
                G_s[L][:,o_r,o_c]=best_match

    # showImagesHorizontally(G_s)
    # print()

    final_output = G_s[-1]

    out_im = image_utils.tensor_to_image(final_output,image_size=out_size,denorm=False)
    out_im.save(output_path)
    return final_output


def main():
    args = args_.parse_arguments()
    n_lvls=4

    large_pics = [
    '2.png',
    '3.png',
    '9.png',
    '12.png',
    '15.png',
    '16.png',
    '19.png',
    '28.png',
    '30.png',
    '31.png',
    ]

    inputs_dir = './inputs/style_images/fdf_textures'
    outputs_dir = args.output_dir

    output_dir = os.path.join(outputs_dir,str(seeder.SEED),'tvsq')
    image_utils.make_dir(output_dir)
    for im in os.listdir(inputs_dir):
        start = timer()
        input_path = os.path.join(inputs_dir,im)
        if os.path.isdir(input_path): continue

        # if im in large_pics:
        #     n_size=64
        # else:
        #     n_size=21
        n_size=35
        # if im == '4.png':
        # filename = f'{im[:-4]}_{n_size}_alt_pyramid_{n_lvls}lvls.png'
        filename = im
        output_path = os.path.join(output_dir,filename)      
        tvsq(input_path,output_path,n_size=n_size,n_levels=n_lvls,in_size=256,out_size=256)
        
        end=timer()
        print(f'Time elapsed for {im}: {end-start:.2f}')
    return


if __name__ == "__main__":   
    main()
    # i_path = './inputs/style_images/fdf_textures/0.png'
    # o_path = './outputs/output_images/TVSQ Run 1/32/tvsq/0_21_alt_pyramid_4lvls.png'
    # o_path_64 = './outputs/output_images/TVSQ Run 1 - 128px/32/tvsq/0_21_alt_pyramid_4lvls.png'
    # input_128 = image_utils.image_to_tensor(image_utils.load_image(i_path,'RGB'),image_size=D.IMSIZE.get()//2).unsqueeze(0)
    # input_256 = image_utils.image_to_tensor(image_utils.load_image(i_path,'RGB'),image_size=D.IMSIZE.get()).unsqueeze(0)
    # output_128 = image_utils.image_to_tensor(image_utils.load_image(o_path_64,'RGB'),image_size=D.IMSIZE.get()//2).unsqueeze(0)
    # output_256 = image_utils.image_to_tensor(image_utils.load_image(o_path,'RGB'),image_size=D.IMSIZE.get()).unsqueeze(0)

    # dist_128 = get_wassdist(input_128,output_128)
    # dist_256 = get_wassdist(input_256,output_256)
    # print(f'FID at 128px size: {dist_128}')
    # print(f'FID at 256px size: {dist_256}')