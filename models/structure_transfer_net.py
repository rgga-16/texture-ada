from numpy.core import numeric
import torch
from torch import nn
import numpy as np
from matplotlib import pyplot as plt
from torch.nn import functional as F
import torchvision
from defaults import DEFAULTS as D
from ops import adaptive_instance_normalization,adain_pointcloud
from helpers.visualizer import display_pointcloud
from models.pointnet import Transformer
from models.networks.vgg import VGG19


import cv2

class GraphProjection(nn.Module):
    """Graph Projection layer, which pool 2D features to mesh

    The layer projects a vertex of the mesh to the 2D image and use 
    bilinear interpolation to get the corresponding feature.
    """

    def __init__(self,mesh_pos=[0., 0., -0.8],camera_f=[128, 128],camera_c=[128, 128],bound=0):
        """
        :param mesh_pos: [x,y,z] positions of mesh
        :param camera_f: 
        :param camera_c: 
        :param bound: 
        """
        super(GraphProjection, self).__init__()
        self.mesh_pos = mesh_pos # mesh_pos =[0., 0., -0.8]
        self.camera_f = camera_f # camera_f = [248., 248.]
        self.camera_c = camera_c # camer_c = [111.5, 111.5]
        self.bound=bound 

    def bound_val(self, x):
        """
        given x, return min(threshold, x), in case threshold is not None
        """
        if self.bound < 0:
            return -self.threshold(-x)
        elif self.bound > 0:
            return self.threshold(x)
        return x
    
    def project(self, img_feature,sample_points):
        """
        :param img_shape: raw image shape
        :param img_feature: [batch_size x channel x h x w] batch of image features
        :param sample_points: [batch_size x num_points x 2], in range [-1, 1]
        :return: [batch_size x feat_dim x num_points]
        """
        sample_points_ = sample_points.unsqueeze(1)
        output = F.grid_sample(img_feature,sample_points_,align_corners=True)
        output = output.squeeze(2)

        return output

    def flatten(self, img_res, shape_verts):
        assert shape_verts.dim()==3
        shape_verts = shape_verts.permute(0,2,1)
        
        half_res = (img_res-1)/2
        camera_c_offset = np.array(self.camera_c) - half_res 

        # map to [-1, 1]
        # not sure why they render to negative x
        # Moves shape verts by mesh_pos
        positions = shape_verts.permute(0,2,1) + torch.tensor(self.mesh_pos, device=shape_verts.device, dtype=torch.float)

        # Tensorflow (original) version: h = 248 * tf.divide(-Y, -Z) + 112
        y_pos = positions[:,:, 1]
        z_pos = self.bound_val(positions[:,:, 2])
        h = self.camera_f[1] * (-y_pos / -z_pos) + camera_c_offset[1]
        # Tensorflow (original) version: w = 248.0 * tf.divide(X, -Z) + 112.0
        x_pos = positions[:,:, 0]
        w = self.camera_f[0] * (x_pos / -z_pos) + camera_c_offset[0]
        
        # directly do clamping. 
        w /= img_res[0]
        h /= img_res[1]

        # normalize between [0,1]
        w =  (w - torch.min(w,dim=1,keepdim=True)[0])/(torch.max(w,dim=1,keepdim=True)[0] - torch.min(w,dim=1,keepdim=True)[0])
        h =  (h - torch.min(h,dim=1,keepdim=True)[0])/(torch.max(h,dim=1,keepdim=True)[0] - torch.min(h,dim=1,keepdim=True)[0])

        # clamp to [0, 1]
        w = torch.clamp(w, min=-1, max=1)
        h = torch.clamp(h, min=-1, max=1)

        # feats = [shape_verts]
        # for img_feature in img_features:
        pcd_xy = torch.stack([w,h],dim=-1)
        # feats.append(self.project(img_feature,pcd_xy).permute(0,2,1))
        # output = torch.cat(feats,1)
        
        # pcd_xy is a flattened version of the point cloud. Flat along the z-axis (front view)
        # pcd_xy = torch.stack([w,h],dim=-1)
        output = pcd_xy
        return output
    
    def forward(self, img_res, img_features, shape_verts):
        """
        :param img_res: [h,w] of image.
        :param img_features: (n_encoder_layers, batch_size x channel x h x w) list of feature map sets
        :param shape_verts: [batch_size,n_verts,3], shape coordinates
        :return: [batch_size,n_verts,3+ total_num_img_featmaps] output, mesh features concatenated with projected image features
        """

        half_res = (img_res-1)/2
        camera_c_offset = np.array(self.camera_c) - half_res 

        # map to [-1, 1]
        # not sure why they render to negative x
        # Moves shape verts by mesh_pos
        positions = shape_verts.permute(0,2,1) + torch.tensor(self.mesh_pos, device=shape_verts.device, dtype=torch.float)

        # Tensorflow (original) version: h = 248 * tf.divide(-Y, -Z) + 112
        y_pos = positions[:,:, 1]
        z_pos = self.bound_val(positions[:,:, 2])
        h = self.camera_f[1] * (-y_pos / -z_pos) + camera_c_offset[1]
        # Tensorflow (original) version: w = 248.0 * tf.divide(X, -Z) + 112.0
        x_pos = positions[:,:, 0]
        w = self.camera_f[0] * (x_pos / -z_pos) + camera_c_offset[0]
        
        # directly do clamping. 
        w /= img_res[0]
        h /= img_res[1]

        # normalize between [0,1]
        w =  (w - torch.min(w,dim=1,keepdim=True)[0])/(torch.max(w,dim=1,keepdim=True)[0] - torch.min(w,dim=1,keepdim=True)[0])
        h =  (h - torch.min(h,dim=1,keepdim=True)[0])/(torch.max(h,dim=1,keepdim=True)[0] - torch.min(h,dim=1,keepdim=True)[0])

        # clamp to [-1, 1]
        w = torch.clamp(w, min=-1, max=1)
        h = torch.clamp(h, min=-1, max=1)

        # feats = [shape_verts]
        # for img_feature in img_features:
        pcd_xy = torch.stack([w,h],dim=-1)
        output = self.project(img_features,pcd_xy)
        # feats.append(self.project(img_feature,pcd_xy).permute(0,2,1))
        # output = torch.cat(feats,1)
        
        # pcd_xy is a flattened version of the point cloud. Flat along the z-axis (front view)
        # pcd_xy = torch.stack([w,h],dim=-1)
        # output = pcd_xy.permute(0,2,1)
        return output

class Pointnet_Autoencoder2(nn.Module):
    def __init__(self,n_points,point_dim=3):
        super(Pointnet_Autoencoder2,self).__init__()
        self.n_points=n_points
        self.hidden_dim = 256

        # self.conv1 = nn.Conv1d(point_dim,32,kernel_size=1,stride=1)
        # self.conv2 = nn.Conv1d(32,64,kernel_size=1,stride=1)
        # # Do projection
        # self.conv3 = nn.Conv1d(128,128,kernel_size=1,stride=1)
        # # Do projection
        # self.conv4 = nn.Conv1d(256, 256, kernel_size=1,stride=1)
        # # Do projection
        # self.conv5 = nn.Conv1d(512, 512, kernel_size=1,stride=1)

        self.conv1 = nn.Conv1d(point_dim,32,kernel_size=1,stride=1)
        self.conv2 = nn.Conv1d(32,64,kernel_size=1,stride=1)
        self.conv3 = nn.Conv1d(64,128,kernel_size=1,stride=1)
        self.conv4 = nn.Conv1d(128, 256, kernel_size=1,stride=1)
        self.conv5 = nn.Conv1d(256, 512, kernel_size=1,stride=1)

        self.deconv5 = nn.ConvTranspose1d(512,256,kernel_size=1,stride=1)
        self.deconv4 = nn.ConvTranspose1d(256,128,kernel_size=1,stride=1)
        self.deconv3 = nn.ConvTranspose1d(128,64,kernel_size=1,stride=1)
        self.deconv2 = nn.ConvTranspose1d(64,32,kernel_size=1,stride=1)
        self.deconv1 = nn.ConvTranspose1d(32,point_dim,kernel_size=1,stride=1)

        self.graph_projection = GraphProjection(mesh_pos=[0.0, 0.0, -0.8],camera_c=[128.0,128.0],camera_f=[128,128])
        

    def forward(self,pointcloud,image_feats):
        # Change pointcloud to shape (batch_size,3,n_points)
        pointcloud = pointcloud.permute(0,2,1)
        assert pointcloud.dim()==3

        x = F.relu(self.conv1(pointcloud))
        pfeat_relu1_2 = F.relu(self.conv2(x)) 
        imfeat_relu1_2 = F.relu(self.graph_projection(np.array([256,256]), image_feats['relu1_2'],pointcloud))
        # pfeat_relu1_2=torch.cat((pfeat_relu1_2,imfeat_relu1_2),dim=1)
        # pfeat_relu1_2 = adain_pointcloud(pfeat_relu1_2,image_feats['relu1_2'])
        pfeat_relu1_2 = adain_pointcloud(pfeat_relu1_2,imfeat_relu1_2)
        
        
        pfeat_relu2_2 = F.relu(self.conv3(pfeat_relu1_2))
        imfeat_relu2_2 = F.relu(self.graph_projection(np.array([256,256]), image_feats['relu2_2'],pointcloud))
        # pfeat_relu2_2=torch.cat((pfeat_relu2_2,imfeat_relu2_2),dim=1)
        # pfeat_relu2_2 = adain_pointcloud(pfeat_relu2_2,image_feats['relu2_2'])
        pfeat_relu2_2 = adain_pointcloud(pfeat_relu2_2,imfeat_relu2_2)
        

        pfeat_relu3_4 = F.relu(self.conv4(pfeat_relu2_2))
        imfeat_relu3_4 = F.relu(self.graph_projection(np.array([256,256]), image_feats['relu3_4'],pointcloud))
        # pfeat_relu3_4=torch.cat((pfeat_relu3_4,imfeat_relu3_4),dim=1)
        # pfeat_relu3_4 = adain_pointcloud(pfeat_relu3_4,image_feats['relu3_4'])
        pfeat_relu3_4 = adain_pointcloud(pfeat_relu3_4,imfeat_relu3_4)
        

        pfeat_relu4_4 = F.relu(self.conv5(pfeat_relu3_4)) 
        # imfeat_relu4_4 = F.relu(self.graph_projection(np.array([256,256]), image_feats['relu4_4'],pointcloud))
        # pfeat_relu4_4=torch.cat((pfeat_relu4_4,imfeat_relu4_4),dim=1)
        # pfeat_relu4_4 = adain_pointcloud(pfeat_relu4_4,image_feats['relu4_4'])
        
        x = F.relu(self.deconv5(pfeat_relu4_4))
        x = F.relu(self.deconv4(x))
        x = F.relu(self.deconv3(x))
        x = F.relu(self.deconv2(x))
        output =self.deconv1(x)
        output =  (output - torch.min(output))/(torch.max(output) - torch.min(output))

        output = output.permute(0,2,1)
        return output




class Pointnet_Autoencoder3(nn.Module):
    def __init__(self,n_points,point_dim=3):
        super(Pointnet_Autoencoder3,self).__init__()
        self.n_points = n_points 
        self.point_dim = point_dim
        self.image_encoder = VGG19()
        self.pointcloud_encoder = Transformer()

        self.flatten = nn.Flatten()
        self.im_linear = nn.Linear(self.imfeat_dim,(self.n_points*self.point_dim))
        self.pcd_linear = nn.Linear(1024, (self.n_points*self.point_dim))

    def forward(self,pointcloud,image):
        # Change pointcloud to shape (batch_size,3,n_points)
        assert pointcloud.dim()==3
        pointcloud = pointcloud.permute(0,2,1)

        image_feats = self.image_encoder(image,layers={'28' : 'conv5_1'})['conv5_1']
        image_feats = self.flatten(image_feats)
        image_feats = self.im_linear(image_feats)
        image_feats = nn.Linear(1024,512)
        
        
        pcd_feat,_,_ = self.pointcloud_encoder(pointcloud)
        pcd_feat = self.pcd_linear(pcd_feat)
        # norm_feat = adain_pointcloud(pcd_feat.unsqueeze(-1),image_feats.unsqueeze(-1))
        output = pcd_feat.view(-1,self.point_dim,self.n_points)
        output = torch.tanh(output)/2.0

        output = output.permute(0,2,1)
        return output





