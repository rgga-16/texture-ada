
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils import data
import torchvision

import numpy as np
import math
from matplotlib import pyplot as plt
import time

import args as args_

from seeder import SEED, init_fn
from helpers import visualizer, logger, model_utils,image_utils
from models import pixel2mesh
from models.networks.vgg import VGG19
from models3d_dataset import Pix3D, Pix3D_Paired
import models.pointnet
from models.pointnet import Pointnet_Autoencoder, Pointnet_UpconvAutoencoder
from models.structure_transfer_net import Pointnet_Autoencoder2, Pointnet_Autoencoder3
from defaults import DEFAULTS as D
import pickle

import copy, datetime, random,os
import kaolin as kal
import open3d as o3d

import multiprocessing
from helpers import logger

args = args_.parse_arguments()

assert args.num_points is not None
n_points=args.num_points

images_dir = './data/filipino_designer_furniture/chairs/masked'
masks_dir = './data/filipino_designer_furniture/chairs/masks'
images = [
    'cobonpue-8-1.png', #red flower
    '35.ch-vito-ale-hi-natwh-1.png', #rectangular basket swing thing
    'cobonpue-92-1.png', #twirled chair
    'selma-19.png', #semicircle folding chair,
    'selma-88.png', #circular chair
]

shapenet_dir = './inputs/3d_models/shapenet/chairs'
model_dirs = [ 
    'armchair_5',
    'dining_chair_1',
    'lounge_sofa_2',
    'office_chair_3'
]
for alpha in [0.2,0.5,0.7,0.9]:
    output_dir = f'./outputs/output_models/[{D.DATE()}] style {alpha}'
    for imfile in images:
        # Create output folder
        # This will store the model, output images, loss history chart and configurations log
        output_folder = os.path.join(output_dir, imfile)
        try:
            os.makedirs(output_folder)
        except FileExistsError:
            pass
        image_path= os.path.join(images_dir,imfile)
        mask_path = os.path.join(masks_dir,imfile)
        for model_dir in model_dirs:
            model_path= os.path.join(shapenet_dir,model_dir,'archive','model.obj')

            # Preprocess model
            verts,faces,_ = model_utils.load_mesh(model_path)
            pcd1 = model_utils.mesh_to_pointcloud(verts,faces,n_points)

            img = image_utils.load_image(image_path)
            mask = image_utils.load_image(mask_path,mode='L')
            masked_img = img 
            masked_img.putalpha(mask)

            # Preprocess image
            im2 = image_utils.image_to_tensor(masked_img,phase='test',image_size=256)[:3,...].expand(1,-1,-1,-1)
            mask2 = image_utils.image_to_tensor(mask,phase='test',image_size=256,normalize=False).expand(1,-1,-1,-1)

            # Setup model
            model = Pointnet_Autoencoder2(n_points=n_points).to(D.DEVICE())

            # Setup Image Feature Extractor
            img_feat_extractor = VGG19()

            # Training algorithm
            lr,n_epochs = args.lr, args.epochs

            optimizer = torch.optim.Adam(model.parameters(),lr=lr)
            pcd1 = pcd1.to(D.DEVICE())
            im2 = im2.to(D.DEVICE())
            mask2 = mask2.to(D.DEVICE())

            #sample points over mask
            segs=[]
            m=mask2[0]
            seg=(m>0.1).nonzero().float().unsqueeze(0).to(D.DEVICE())
            if seg.shape[1]>n_points:
                seg_idx = torch.randperm(seg.shape[1])
                seg_idx = seg_idx[:n_points]
                seg = seg[:,seg_idx,:]
            segs.append(seg)
            segs=torch.cat(segs)
            segs=segs[:,:,1:]
            third_dim = torch.zeros_like(segs,device=D.DEVICE())[:,:,-1].unsqueeze(-1)
            segs=torch.cat((segs,third_dim),dim=-1 )
            segs=segs.detach()
            segs/=mask2.shape[-1]

            since = int(round(time.time()*1000))
            for epoch in range(n_epochs):
                model.train()
                train_running_loss=0.0
                train_loss,val_loss=0.0,0.0

                optimizer.zero_grad()
                image_feats = img_feat_extractor(im2,layers=D.STYLE_LAYERS.get())
                
                output,output_masks = model(pcd1,image_feats,im2)
                
                third_dim = torch.zeros((output_masks.shape[0],output_masks.shape[1],1),device=D.DEVICE())
                output_masks=torch.cat((output_masks,third_dim),dim=-1)


                style_loss = models.pointnet.pointcloud_autoencoder_loss(output_masks,segs)
                content_loss = models.pointnet.pointcloud_autoencoder_loss(output,pcd1)
                style_weight = alpha
                content_weight = 1-style_weight
                loss = (style_loss*style_weight) + (content_loss*content_weight)
                loss.backward()
                optimizer.step()
                
                train_loss+= loss.item() * 1
                train_running_loss+=loss.item()
                
                print('\n[Epoch {}]\t Train Loss: {}\t \n'.format(epoch+1,train_loss))
                # if (epoch%epoch_chkpt==epoch_chkpt-1):
                #     gen_path = f'{model.__class__.__name__}_chkpt.pth'
                #     torch.save(model.state_dict(),gen_path)
                #     print(f'[Epoch {epoch+1}]\t Model saved in {gen_path}\n')

            model_file = f'{imfile[:-4]}_{model_dir}.pth'
            gen_path = os.path.join(output_folder,model_file)
            torch.save(model.state_dict(),gen_path)
            print(f'Final Model saved in {gen_path}\n')

            model.load_state_dict(torch.load(gen_path))

            # Visualize test pointclouds
            model.eval()
            with torch.no_grad():
                image_feats = img_feat_extractor(im2,layers=D.STYLE_LAYERS.get())
                final_output,output_mask = model(pcd1,image_feats,im2)

            output_mask=torch.cat((output_mask,third_dim),dim=-1)
            final_output = model_utils.NormalizePointcloud()(final_output)

            gen_pcd = model_utils.pointcloud_kaolin_to_open3d(final_output)
            real_pcd = model_utils.pointcloud_kaolin_to_open3d(pcd1)
            gen_mesh = model_utils.pointcloud_to_mesh_poisson(gen_pcd,depth=5)

            gen_mesh_file = f'{model_dir}.obj'
            gen_pcd_file = f'{model_dir}.ply'
            real_pcd_file = f'{model_dir}_real.ply'
            o3d.io.write_triangle_mesh(os.path.join(output_folder,gen_mesh_file),gen_mesh)
            o3d.io.write_point_cloud(os.path.join(output_folder,gen_pcd_file), gen_pcd)
            o3d.io.write_point_cloud(os.path.join(output_folder,real_pcd_file), real_pcd)

