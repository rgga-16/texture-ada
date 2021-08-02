'''
This file contains non-ML methods for texture synthesis
'''
import torch
from torch.nn import MSELoss
from defaults import DEFAULTS as D

from itertools import product
import seeder
from seeder import SEED
from helpers import image_utils
import ops, style_transfer as st

from operations.texture_synthesis.image_quilting import image_quilting
from operations.texture_synthesis.tree_struct_vector_quantization import tsvq

import random, numpy as np, os

def log_dists(path,dists):
    with open(path,'w') as f:
        f.writelines('\n'.join(dists))
    return

def get_wassdist(ref_texture,output_texture):

    style_means,style_covs = st.get_means_and_covs(ref_texture)
    output_means,output_covs = st.get_means_and_covs(output_texture)

    wass_dist = 0
    for s in D.STYLE_LAYERS.get().values():
        wdist = ops.gaussian_wasserstein_distance(style_means[s],style_covs[s],output_means[s],output_covs[s]).real
        wass_dist += D.SL_WEIGHTS.get()[s] * wdist 
    wasserstein_distance = torch.mean(wass_dist)
    
    return wasserstein_distance.item()

if __name__ == "__main__":   


    
    seeder.set_seed(SEED)
    inputs_dir = './inputs/style_images/filipino_designer_furniture_textures'
    image_quilt_dir = os.path.join(inputs_dir,'synthetic/Non-DL Methods/Image Quilting')
    tree_structured_dir = os.path.join(inputs_dir,'synthetic/Non-DL Methods/Tree Structured Vector Quantization')
    
    output_size=D.IMSIZE.get()

    texture_synth_dirs = [image_quilt_dir,tree_structured_dir]
    for seed in [23,64,128,256]:
        seeder.set_seed(seed)
        for tsdir in texture_synth_dirs:
            output_dir = os.path.join(tsdir,str(seed))
            image_utils.make_dir(output_dir)
            wassdists=[] 
            log = []
            method_name = os.path.basename(tsdir)
            for im in os.listdir(inputs_dir):
                ref_path = os.path.join(inputs_dir,im)
                if os.path.isdir(ref_path): continue
                output_path = os.path.join(output_dir,im)

                if method_name=='Image Quilting':
                    image_quilting(ref_path,output_path,(output_size,output_size),32,overlap=0,tolerance=0.1)
                elif method_name=='Tree Structured Vector Quantization':
                    tsvq(ref_path,output_path)
                else:
                    raise Exception('Method not found')
                
                ref_txture = image_utils.image_to_tensor(image_utils.load_image(ref_path,'RGB'),normalize=False).unsqueeze(0)
                output_txture = image_utils.image_to_tensor(image_utils.load_image(output_path,'RGB'),normalize=False).unsqueeze(0)
                wassdist = get_wassdist(ref_txture,output_txture)
                wassdists.append(wassdist)
                logged_dist=f'{im} : {wassdist:.8f}'
                print(logged_dist)
                log.append(logged_dist)
            
            mean_dist = np.mean(np.array(wassdists))
            mean_log = f'{method_name} - Average Wass Dist: {mean_dist:.8f}'
            print(mean_log)
            log.append(mean_log)
            log_dists(os.path.join(output_dir,'dists.txt'),log)
                
                
    # iq_wassdists=[]
    # iq_log=[]
    # for im in os.listdir(inputs_dir):
    #     ref_path = os.path.join(inputs_dir,im)
    #     if os.path.isdir(ref_path): continue
    #     output_path = os.path.join(image_quilt_dir,im)

    #     image_quilting(ref_path,output_path,(output_size,output_size),32,overlap=0,tolerance=0.1)

    #     ref_txture = image_utils.image_to_tensor(image_utils.load_image(ref_path,'RGB'),normalize=False).unsqueeze(0)
    #     output_txture = image_utils.image_to_tensor(image_utils.load_image(output_path,'RGB'),normalize=False).unsqueeze(0)
    #     wassdist = get_wassdist(ref_txture,output_txture)
    #     iq_wassdists.append(wassdist)
    #     logged_dist=f'{im} : {wassdist:.8f}'
    #     print(logged_dist)
    #     iq_log.append(logged_dist)

    # mean_iq_dist=np.mean(np.array(iq_wassdists))
    # mean_log = f'Image Quilting - Average Wass Dist: {mean_iq_dist:.8f}'
    # print(mean_log)
    # iq_log.append(mean_log)
    # log_dists(os.path.join(image_quilt_dir,'dists.txt'),iq_log)

    # ts_wassdists=[]
    # ts_log=[]
    # for im in os.listdir(inputs_dir):
    #     ref_path = os.path.join(inputs_dir,im)
    #     if os.path.isdir(ref_path): continue
    #     output_path = os.path.join(tree_structured_dir,im)
        
        
    #     tsvq(ref_path,output_path)
    #     ref_txture = image_utils.image_to_tensor(image_utils.load_image(ref_path,'RGB'),normalize=False).unsqueeze(0)
    #     output_txture = image_utils.image_to_tensor(image_utils.load_image(output_path,'RGB'),normalize=False).unsqueeze(0)
    #     wassdist = get_wassdist(ref_txture,output_txture)
    #     ts_wassdists.append(wassdist)
    #     logged_dist= f'{im} : {wassdist:.8f}'
    #     print(logged_dist)
    #     ts_log.append(logged_dist)

    # mean_ts_dist = np.mean(np.array(ts_wassdists))
    # mean_log = f'Tree Structured - Average Wass Dist: {mean_ts_dist:.8f}'
    # print(mean_log)
    # ts_log.append(mean_log)
    # log_dists(os.path.join(tree_structured_dir,'dists.txt'),ts_log)

    