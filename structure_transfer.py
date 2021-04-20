
import torch
from torch import nn
from torch._C import device 
from torch.nn import functional as F
from torch.utils import data
import torchvision

import numpy as np
import math
from matplotlib import pyplot as plt
import time

from args import args
from helpers import visualizer
from models import structure_transfer_models, texture_transfer_models
from dataset import Pix3D
import models.pointnet
from models.pointnet import Pointnet_Autoencoder
from defaults import DEFAULTS as D
import pickle
from helpers import logger
torch.manual_seed(0)

import copy, datetime
import kaolin as kal
import random

from scipy.sparse import coo_matrix

def load_dat(filename):
    with open(filename,"rb") as fp:
        fp_info = pickle.load(fp,encoding='latin1')

        print()
    
    return fp_info


SEED=5
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)  # if you are using multi-GPU.
np.random.seed(SEED)  # Numpy module.
random.seed(SEED)  # Python random module.
torch.manual_seed(SEED)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

def _init_fn(workder_id):
    np.random.seed(int(SEED))

n_points=2048

# Setup dataset
##########################################################
dataset= Pix3D(n_points,category='chair')

train_size = int(0.75 * dataset.__len__())
test_size = dataset.__len__() - train_size 

train_dataset, test_dataset = data.random_split(dataset,[train_size,test_size],
                                                            generator = torch.Generator().manual_seed(SEED))

train_loader = data.DataLoader(train_dataset,batch_size=16,worker_init_fn=_init_fn)
test_loader = data.DataLoader(test_dataset,batch_size=16,worker_init_fn=_init_fn)
##########################################################

# Setup model
##########################################################
model = Pointnet_Autoencoder(n_points=n_points).to(D.DEVICE())

##########################################################

# Training algorithm
##########################################################
lr = 0.0001
n_epochs = 3
checkpoint = 4


optimizer = torch.optim.Adam(model.parameters(),lr=lr)
since = int(round(time.time()*1000))
print('Start training')
for epoch in range(n_epochs):
    # print('Epoch {}'.format(epoch+1))
    # print('='*10)
    model.train()
    running_loss=0.0
    for i, data_ in enumerate(train_loader,0):
        pointcloud,image = data_
        pointcloud = pointcloud.to(D.DEVICE())
        image = image.to(D.DEVICE())

        optimizer.zero_grad()

        output = model(pointcloud)
        batch_size = pointcloud.shape[0]
        loss = models.pointnet.pointcloud_autoencoder_loss(output,pointcloud,batch_size)
    
        loss.backward()
        optimizer.step()

        running_loss+=loss.item()
        if(i%checkpoint==checkpoint-1):
            print('[Epoch {} | Batch {}/{} ] Loss: {:.3f}'.format(epoch+1,i+1,len(train_loader),running_loss/checkpoint))
            running_loss=0.0
time_elapsed = int(round(time.time()*1000)) - since
print ('training time elapsed {}ms'.format(time_elapsed))

model_file = '{}.pth'.format(model.__class__.__name__)
gen_path = model_file
torch.save(model.state_dict(),gen_path)
print('Model saved in {}'.format(gen_path))



# Visualize final output
##########################################################

pointclouds,images = iter(test_loader).next() #use train_loader to see if network overfits train data
test_pointcloud = pointclouds[:2]

model.eval()
with torch.no_grad():
    final_output = model(test_pointcloud)


# Visualizing the final pointcloud
for tp, f in zip(test_pointcloud,final_output):
    visualizer.display_pointcloud(tp)
    visualizer.display_pointcloud(f)


# mesh_file = './inputs/shape_samples/armchair/model.obj'

# Visualizing the mesh and visualizing its vertices only
##########################################################
# x,y,z = np.array(vertices).T
# i,j,k = np.array(faces).T

# # Visualize mesh
# go.Figure(data=[go.Mesh3d(x=x,y=z,z=y,
#                         opacity=0.5,
#                         i=i,j=j,k=k)]).show()
# # Visualize vertices only
# # pcshow(x,y,z)
# scatter =go.Figure(data=[go.Scatter3d(x=x, y=y, z=z,mode='markers')])
# scatter.update_traces(marker=dict(size=2,
#                       line=dict(width=2,
#                       color='DarkSlateGrey')),
#                       selector=dict(mode='markers')).show()
##########################################################

# Sample points over mesh to retrieve a pointcloud
##########################################################
# n_samples = 3000
# raw_pointcloud, selected_faces_idx = kal.ops.mesh.sample_points(vertices.unsqueeze(0),
#                                                             faces,num_samples=n_samples)
# raw_pointcloud.squeeze_(0)    # pointcloud = (n_samples,3)
##########################################################

# Visualizing the pointcloud
##########################################################
# x,y,z = np.array(pointcloud.squeeze()).T
# # Visualize pointcloud
# scatter =go.Figure(data=[go.Scatter3d(x=x, y=y, z=z,mode='markers')])
# scatter.update_traces(marker=dict(size=2,
#                       line=dict(width=2,
#                       color='DarkSlateGrey')),
#                       selector=dict(mode='markers')).show()
##########################################################

# Augment the pointcloud
##########################################################
# Normalize pointcloud
# pointcloud_norm = raw_pointcloud - torch.mean(raw_pointcloud,dim=0)
# pointcloud_norm /= torch.max(torch.linalg.norm(pointcloud_norm,dim=1))

# # Rotate pointcloud along z-axis
# theta = random.random()*2.0 * math.pi 
# rot_matrix = torch.tensor([[math.cos(theta),-math.sin(theta),0],
#                            [math.sin(theta),math.cos(theta),0], 
#                            [0,0,1]])

# rot_pointcloud = torch.matmul(pointcloud_norm,rot_matrix)

# # Add noise into pointcloud
# noise = torch.normal(mean=0.0,std=0.02,size=raw_pointcloud.shape)

# noisy_pointcloud = rot_pointcloud+noise

# pointcloud = noisy_pointcloud.clone()
# # pointcloud_tensor = pointcloud.view(pointcloud.shape[1],pointcloud.shape[0]).unsqueeze(0)

# # Visualizing the augmented pointcloud
# x,y,z = np.array(pointcloud.squeeze()).T
# scatter =go.Figure(data=[go.Scatter3d(x=x, y=y, z=z,mode='markers')])
# scatter.update_traces(marker=dict(size=2,
#                       line=dict(width=2,
#                       color='DarkSlateGrey')),
#                       selector=dict(mode='markers')).show()

# pointcloud = pointcloud.to(D.DEVICE())
##########################################################







# #####################################################

# class Ellipsoid(object):

#     def __init__(self,filepath,mesh_pos= [0., 0., -0.8]):
#         fp_info = load_dat(filepath)

#         # verts = (n_verts,3)
#         self.verts = torch.tensor(fp_info[0]) - torch.tensor(mesh_pos,dtype=torch.float)
#         self.verts = self.verts.to(D.DEVICE())

#         # edges & faces & lap_idx
#         # edge = (num_edges,2)
#         # faces = (num_faces,4)
#         # laplace_idx = (num_pts,10)
#         self.edges, self.laplace_idx = [], []
#         for i in range(3):
#             self.edges.append(torch.tensor(fp_info[1 + i][1][0], dtype=torch.long).to(D.DEVICE()))
#             self.laplace_idx.append(torch.tensor(fp_info[7][i], dtype=torch.long).to(D.DEVICE()))
        
#         # unpool index
#         # un_pool = (num_pool_edges 2)
#         # unpool_idx[0] = (462,2), 
#         # unpool_idx[1] = (1848,2)
#         self.unpool_idx = [torch.tensor(fp_info[4][i], dtype=torch.long).to(D.DEVICE()) for i in range(2)]

#         # loops and adjacent edges (adjacency matrix)
#         self.adj_mat = []
#         for i in range(1, 4):
#             # Returned as 3 lists where:
#             # 0: np.array, 2D, pos
#             # 1: np.array, 1D, vals
#             # 2: tuple - shape, n * n
#             adj_mat_init = fp_info[i][1]
#             # Converts to a sparse tensor
#             adj_mat_ = torch_sparse_tensor(*adj_mat_init)
#             self.adj_mat.append(adj_mat_)
        

#         ellipsoid_dir = os.path.dirname(filepath)
#         self.faces = []
#         self.obj_fmt_faces = []
#         # faces: f * 3, original ellipsoid, and two after deformations
#         for i in range(1, 4):
#             face_file = os.path.join(ellipsoid_dir, "face%d.obj" % i)
#             faces_txt = np.loadtxt(face_file, dtype='|S32')
#             self.obj_fmt_faces.append(faces_txt)
#             faces_ = torch.tensor(faces_txt[:, 1:].astype(np.int) - 1).to(D.DEVICE())
#             self.faces.append(faces_)

# filename = './inputs/ellipsoid/info_ellipsoid.dat'

# ellipsoid = Ellipsoid(filename)

# print()

# tensor = utils.image_to_tensor(utils.load_image('./inputs/style_images/tiled/chair-3_tiled.png'),image_size=224)
# tensor = tensor[:3,...].unsqueeze(0)
# b,c,h,w = tensor.shape

# encoder = structure_transfer_models.VGG16_Encoder(3)

# image_feat_vgg16_layers = {
#     '4':'maxpool_1', # 1,64,112,112 feat maps
#     '9':'maxpool_2', # 1,128,56,56 feat maps
#     '16':'maxpool_3', # 1,256,28,28 feat maps
#     '23':'maxpool_4', # 1,512,14,14 feat maps
#     # '30':'maxpool_5', # 1,512,7,7 feat maps
# }

# image_feats = encoder(tensor,layers=image_feat_vgg16_layers)
# image_feats = list(image_feats.values())
# num_image_feats = np.sum([feats.shape[1] for feats in image_feats])

# encoder2 = structure_models.VGG16P2M(3).to(D.DEVICE()).eval()

# 1,64,56,56 feat maps
# 1,128,28,28 feat maps
# 1,256,14,14 feat maps
# 1,512,7,7 feat maps
# image_feats2 = encoder2(tensor)
# num_image_feats_2 = np.sum([feats.shape[1] for feats in image_feats2])
# print()

# p2m = structure_transfer_models.Pixel2MeshModel(init_shape=ellipsoid)
# # print(p2m)

# output_dict = p2m(tensor)
# print()

# #####################################################################
