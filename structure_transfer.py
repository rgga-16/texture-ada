
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils import data
import torchvision

import numpy as np
import math
from matplotlib import pyplot as plt
import time

from args import args
from seeder import SEED, init_fn
from helpers import visualizer, logger, model_utils
from models import structure_transfer_models, texture_transfer_models
from models3d_dataset import Pix3D
import models.pointnet
from models.pointnet import Pointnet_Autoencoder
from defaults import DEFAULTS as D
import pickle

import copy, datetime, random,os
import kaolin as kal
import open3d as o3d

import multiprocessing

assert args.num_points is not None
n_points=args.num_points

# Setup dataset
##########################################################
dataset= Pix3D(n_points,category='chair')
train_size, val_size, test_size = round(0.70 * dataset.__len__()),round(0.20 * dataset.__len__()),round(0.10 * dataset.__len__())
train_dataset, val_dataset, test_dataset = data.random_split(dataset,[train_size,val_size,test_size],
                                                            generator = torch.Generator().manual_seed(SEED))
n_workers = multiprocessing.cpu_count()//2 if args.multiprocess else 0
train_loader = data.DataLoader(train_dataset,batch_size=16,worker_init_fn=init_fn,num_workers=n_workers)
val_loader = data.DataLoader(val_dataset,batch_size=16,worker_init_fn=init_fn,num_workers=n_workers)
test_loader = data.DataLoader(test_dataset,batch_size=16,worker_init_fn=init_fn,num_workers=n_workers)

# Setup model
model = Pointnet_Autoencoder(n_points=n_points).to(D.DEVICE())

# Training algorithm
lr,n_epochs = args.lr, args.epochs
batch_train_chkpt = 1 if len(train_loader) <= args.num_batch_chkpts else len(train_loader)//args.num_batch_chkpts
batch_val_chkpt = 1 if len(val_loader) <= args.num_batch_chkpts else len(val_loader)//args.num_batch_chkpts
epoch_chkpt = 1 if n_epochs <= args.num_epoch_chkpts else n_epochs//args.num_epoch_chkpts

optimizer = torch.optim.Adam(model.parameters(),lr=lr)

since = int(round(time.time()*1000))
print('Start training')
for epoch in range(n_epochs):
    model.train()
    train_running_loss=0.0
    train_loss,val_loss=0.0,0.0
    for i, data_ in enumerate(train_loader,0):
        pointcloud,image = data_
        bs = pointcloud.shape[0]
        pointcloud = pointcloud.to(D.DEVICE())
        image = image.to(D.DEVICE())

        optimizer.zero_grad()

        output = model(pointcloud)

        loss = models.pointnet.pointcloud_autoencoder_loss(output,pointcloud)
        loss.backward()
        optimizer.step()

        train_loss+= loss.item() * bs
        train_running_loss+=loss.item()
        if(i%batch_train_chkpt==batch_train_chkpt-1):
            print('[Epoch {} | Train Batch {}/{} ] Loss: {:.3f}'.format(epoch+1,i+1,len(train_loader),train_running_loss/batch_train_chkpt))
            train_running_loss=0.0

    model.eval()
    val_running_loss=0.0
    for j,data_ in enumerate(val_loader):
        pointcloud,image = data_
        bs = pointcloud.shape[0]
        pointcloud = pointcloud.to(D.DEVICE())
        image = image.to(D.DEVICE())
        # Get output
        with torch.no_grad():
            output = model(pointcloud)
            # Get val loss and other metrics
            loss,f_score,precision,recall = models.pointnet.pointcloud_autoencoder_loss(output,pointcloud,is_eval=True)
        val_loss+= loss.item() * bs
        val_running_loss+=loss.item()
        if(i%batch_val_chkpt==batch_val_chkpt-1):
            print('[Epoch {} | Val Batch {}/{} ] Loss: {:.3f}\t F-Score: {}\t Precision: {}\t Recall: {}'.format(epoch+1,j+1,
            len(val_loader),val_running_loss/batch_val_chkpt,f_score,precision,recall))
            val_running_loss=0.0
    print('\n[Epoch {}]\t Train Loss: {:.3f}\t  Validation Loss: {:.3f}\t \n'.format(epoch+1,
                                                                                train_loss/len(train_loader), 
                                                                                val_loss/len(val_loader)))
    if (epoch%epoch_chkpt==epoch_chkpt-1):
        gen_path = os.path.join(args.output_dir,f'{model.__class__.__name__}_epoch-{epoch+1}.pth')
        torch.save(model.state_dict(),gen_path)
        print(f'[Epoch {epoch+1}]\t Model saved in {gen_path}\n')

time_elapsed = int(round(time.time()*1000)) - since
print ('training time elapsed {}ms'.format(time_elapsed))

model_file = '{}_final.pth'.format(model.__class__.__name__)
gen_path = model_file
torch.save(model.state_dict(),gen_path)
print(f'Final Model saved in {gen_path}\n')

# Visualize test pointclouds
##########################################################
pointclouds,images = iter(test_loader).next() #use train_loader to see if network overfits train data
test_pointcloud = pointclouds[:5]
model.eval()
with torch.no_grad():
    final_output = model(test_pointcloud)

for tp, f in zip(test_pointcloud,final_output):
    visualizer.display_pointcloud(tp)
    visualizer.display_pointcloud(f)





##########################################################

# model_path = './inputs/bunny/reconstruction/bun_zipper.ply'
# # model_path = './inputs/shape_samples/lounge_sofa/archive/model.obj'

# mesh_o3d = o3d.io.read_triangle_mesh(model_path)
# mesh_o3d.compute_vertex_normals()
# # o3d.visualization.draw_geometries([mesh_o3d],width=1280,height=820)

# # Sample points using Open3D
# pcd_uniform = mesh_o3d.sample_points_uniformly(number_of_points=n_points,use_triangle_normal=False,seed=SEED)
# pcd_poisson = mesh_o3d.sample_points_poisson_disk(number_of_points=n_points,use_triangle_normal=False,seed=SEED)
# # o3d.visualization.draw_geometries([pcd_poisson],width=1280,height=820)

# # Sample points using Kaolin then loading into an Open3D object
# verts,faces,_ = model_utils.load_mesh(model_path)
# pcd_kal = model_utils.mesh_to_pointcloud(verts,faces,n_points)
# pcd_o3d = model_utils.pointcloud_kaolin_to_open3d(pcd_kal)

# # Poisson Reconstruction
# mesh_poisson_poisson = model_utils.reconstruct_mesh_poisson(pcd_poisson)
# o3d.visualization.draw_geometries([pcd_poisson,mesh_poisson_poisson],width=1280,height=820)
# o3d.io.write_triangle_mesh("reconstructed_pcd_poisson.obj",mesh_poisson_poisson)
    