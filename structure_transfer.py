
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



# Setup dataset
dataset= Pix3D_Paired(n_points,categories=['chair'],lower_size=50)
train_size, val_size, test_size = round(0.60 * dataset.__len__()),round(0.20 * dataset.__len__()),round(0.20 * dataset.__len__())
train_dataset, val_dataset, test_dataset = data.random_split(dataset,[train_size,val_size,test_size],
                                                            generator = torch.Generator().manual_seed(SEED))
n_workers = multiprocessing.cpu_count()//2 if args.multiprocess else 0
train_loader = data.DataLoader(train_dataset,batch_size=8,worker_init_fn=init_fn,num_workers=n_workers)
val_loader = data.DataLoader(val_dataset,batch_size=8,worker_init_fn=init_fn,num_workers=n_workers)
test_loader = data.DataLoader(test_dataset,batch_size=8,worker_init_fn=init_fn,num_workers=n_workers)

# Setup model
model = Pointnet_Autoencoder3(n_points=n_points).to(D.DEVICE())

# Setup Image Feature Extractor
img_feat_extractor = VGG19()

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
        pointcloud,image,mask,pointcloud2,image2,mask2 = data_
        bs = pointcloud.shape[0]
        pointcloud = pointcloud.to(D.DEVICE())
        image = image.to(D.DEVICE())
        mask = mask.to(D.DEVICE())
        pointcloud2 = pointcloud2.to(D.DEVICE())
        image2 = image2.to(D.DEVICE())
        mask2 = mask2.to(D.DEVICE())

        segs=[]
        for m in mask2:
            seg=(m>0.0).nonzero().float().unsqueeze(0).to(D.DEVICE())
            if seg.shape[1]>n_points:
                seg_idx = torch.randperm(seg.shape[1])
                seg_idx = seg_idx[:n_points]
                seg = seg[:,seg_idx,:]
            segs.append(seg)
        
        segs=torch.cat(segs)

        # image_feats = img_feat_extractor(image2,layers=D.STYLE_LAYERS.get())

        optimizer.zero_grad()
        # visualizer.display_pointcloud(pointcloud[3])
        output,img = model(pointcloud,image2)
        mse_loss = nn.MSELoss()

        mse = mse_loss(img,segs)
        loss = models.pointnet.pointcloud_autoencoder_loss(output,pointcloud2)
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
        pointcloud,image,pointcloud2,image2 = data_
        bs = pointcloud.shape[0]
        pointcloud = pointcloud.to(D.DEVICE())
        image = image.to(D.DEVICE())
        pointcloud2 = pointcloud2.to(D.DEVICE())
        image2 = image2.to(D.DEVICE())

        with torch.no_grad():
            image_feats = img_feat_extractor(image2,layers=D.STYLE_LAYERS.get())
            output = model(pointcloud,image_feats)
            loss,f_score,precision,recall = models.pointnet.pointcloud_autoencoder_loss(output,pointcloud2,is_eval=True)
        
        val_loss+= loss.item() * bs
        val_running_loss+=loss.item()
        if(j%batch_val_chkpt==batch_val_chkpt-1):
            print('[Epoch {} | Val Batch {}/{} ] Loss: {:.3f}\t F-Score: {:.3f}\t Precision: {:.3f}\t Recall: {:.3f}'.format(epoch+1,j+1,
            len(val_loader),val_running_loss/batch_val_chkpt,f_score.mean(),precision.mean(),recall.mean()))
            val_running_loss=0.0
    print('\n[Epoch {}]\t Train Loss: {:.3f}\t  Validation Loss: {:.3f}\t \n'.format(epoch+1,
                                                                                train_loss/len(train_loader), 
                                                                                val_loss/len(val_loader)))
    if (epoch%epoch_chkpt==epoch_chkpt-1):
        gen_path = f'{model.__class__.__name__}_chkpt.pt'
        torch.save(model.state_dict(),gen_path)
        print(f'[Epoch {epoch+1}]\t Model saved in {gen_path}\n')

time_elapsed = int(round(time.time()*1000)) - since
print ('training time elapsed {}ms'.format(time_elapsed))

model_file = '{}_final.pth'.format(model.__class__.__name__)
gen_path = model_file
torch.save(model.state_dict(),gen_path)
print(f'Final Model saved in {gen_path}\n')

model.load_state_dict(torch.load(gen_path))
# Visualize test pointclouds
pointclouds,images = iter(test_loader).next()
test_pointcloud = pointclouds[:5]
test_images = images[:5]
model.eval()
with torch.no_grad():
    image_feats = img_feat_extractor(test_images,layers=D.STYLE_LAYERS.get())
    final_output = model(test_pointcloud,image_feats)

for tp, f in zip(test_pointcloud,final_output):
    visualizer.display_pointcloud(tp)
    visualizer.display_pointcloud(f)

