from io import BufferedIOBase
from numpy.core.fromnumeric import shape
import torch
from torch._C import device
import torch.nn as nn
from torch.nn import Threshold
from torch.nn import functional as F
from torchvision import models, transforms

from defaults import DEFAULTS as D

import math 
import numpy as np

def bmm(matrix,batch):
    return torch.stack([matrix.mm(b) for b in batch], dim=0)

def dot(x,y,is_sparse=False):
    if is_sparse:
        return bmm(x,y)
    else: 
        return torch.matmul(x,y)



class Pixel2MeshModel(nn.Module):
    def __init__(self, init_shape,
                camera_f=[248., 248.],camera_c=[111.5, 111.5],
                mesh_pos=[0., 0., -0.8],
                # 960 = 512 + 256 + 128 + 64
                n_image_feats=960):
        super(Pixel2MeshModel,self).__init__()

        self.init_verts = nn.Parameter(init_shape.verts,requires_grad=False)

        # n_verts, 3
        n_verts,n_coord_channels = self.init_verts.shape
        self.coord_dim = n_coord_channels

        self.hidden_dim = 192
        self.last_hidden_dim = 192

        # 963 features
        self.features_dim = n_image_feats+self.coord_dim

        # self.img_encoder = VGG16_Encoder(3)
        self.img_encoder = VGG16P2M(3).to(D.DEVICE()).eval()

        
        self.gcn_1 = GraphConvBottleneck(6,self.features_dim,self.hidden_dim,
                                        self.coord_dim,init_shape.adj_mat[0])

                                        # 6, 963+192=1155, 192, 3, adj_mat[1]
        self.gcn_2 = GraphConvBottleneck(6,self.features_dim+self.hidden_dim,self.hidden_dim,
                                        self.coord_dim,init_shape.adj_mat[1])

        self.gcn_3 = GraphConvBottleneck(6,self.features_dim+self.hidden_dim,self.hidden_dim,
                                        self.last_hidden_dim,init_shape.adj_mat[2])
        
        self.g_unpooling_1 = GraphUnpoolingLayer(init_shape.unpool_idx[0])
        self.g_unpooling_2 = GraphUnpoolingLayer(init_shape.unpool_idx[1])
        # self.g_unpooling_1 = GraphUnpoolingLayer(init_shape.edges[0])
        # self.g_unpooling_2 = GraphUnpoolingLayer(init_shape.edges[0])
        

        self.projection = GraphProjection(mesh_pos=mesh_pos,
                                        camera_f=camera_f,
                                        camera_c=camera_c)

        self.output_gconv = GraphConvolutionLayer(self.last_hidden_dim,out_feats=self.coord_dim,
                                                adj_mat=init_shape.adj_mat[2])


    
    def forward(self,image):
        # get initial shape coordinates
        # get image_feats = encoder(image)
        batch_size = image.shape[0]
        image_shape = np.array([image.shape[2],image.shape[3]])

        image_feats = self.img_encoder(image)
        init_verts = self.init_verts.data.unsqueeze(0).expand(batch_size,-1,-1)

        # Deform Block 1 
        # projection
        x = self.projection(image_shape,image_feats,init_verts)
        # graph convolution
        x1, x_hidden = self.gcn_1(x)

        # Graph Unpooling 1
        x1_unpooled = self.g_unpooling_1(x1)

        # Deform Block 2
        # projection
        x = self.projection(image_shape,image_feats,x1)
        # graph convolution
        x = self.g_unpooling_1(torch.cat([x,x_hidden],2))
        x2, x_hidden = self.gcn_2(x)

        # Graph Unpooling 2
        x2_unpooled = self.g_unpooling_2(x2)

        # Deform Block 3
        # projection
        x = self.projection(image_shape,image_feats,x2)
        # graph convolution
        x = self.g_unpooling_2(torch.cat([x,x_hidden],2))
        x3, _ = self.gcn_3(x)
        x3 = F.relu(x3)

        # One last graph convolution
        x3 = self.output_gconv(x3)

        return {
            "predicted_vertices": [x1,x2,x3],
            "predicted_vertices_before_conv":[init_verts,x1_unpooled,x2_unpooled]
        }


class GraphConvBottleneck(nn.Module):
    def __init__(self,n_blocks,in_feats,hidden_feats,out_feats,adj_mat):
        super(GraphConvBottleneck,self).__init__()

        self.input_conv = GraphConvolutionLayer(in_feats,hidden_feats,adj_mat)
        
        self.resblocks = nn.Sequential(*[ 
            GraphConvResBlock(hidden_feats,hidden_feats,adj_mat) for i in range(n_blocks)
        ])

        
        self.output_conv = GraphConvolutionLayer(hidden_feats,out_feats,adj_mat)
        self.activation = F.relu
    
    def forward(self,shape_verts):
        """
        :param shape_verts: [batch_size,n_verts,in_feats], shape verts
        :return: [batch_size,n_verts,out_feats] updated shape verts
        """
        x = self.input_conv(shape_verts)
        x = self.activation(x)
        x_hidden = self.resblocks(x)
        x_output = self.output_conv(x_hidden)

        return x_output,x_hidden


# GraphResBlock
class GraphConvResBlock(nn.Module):

    def __init__(self,in_feats,hidden_feats,adj_mat):
        super(GraphConvResBlock,self).__init__()
        self.conv1 = GraphConvolutionLayer(in_feats=in_feats,out_feats=hidden_feats,adj_mat=adj_mat)
        self.conv2 = GraphConvolutionLayer(in_feats=hidden_feats,out_feats=in_feats,adj_mat=adj_mat)
        self.activation = F.relu
    
    def forward(self,shape_verts):
        """
        :param shape_verts: [batch_size,n_verts,3], shape verts
        :return: [batch_size,n_verts,3] updated shape verts = (shape verts + convoluted shape verts)/2
        """
        x = self.conv1(shape_verts)
        x=self.activation(x)
        x = self.conv2(x)
        x=self.activation(x)
        output = (shape_verts+x) * 0.5
        return output

class GraphConvolutionLayer(nn.Module):
    """Simple GCN layer

        Similar to https://arxiv.org/abs/1609.02907
    """
    def __init__(self,in_feats,out_feats,adj_mat):
        super(GraphConvolutionLayer,self).__init__()
        self.in_feats = in_feats
        self.out_feats = out_feats
        self.adj_mat = nn.Parameter(adj_mat,requires_grad=False)

        self.weight = nn.Parameter(torch.Tensor(in_feats,out_feats).to(D.DEVICE()))
        self.loop_weight = nn.Parameter(torch.Tensor(in_feats,out_feats).to(D.DEVICE()))

        self.bias = nn.Parameter(torch.zeros(out_feats).to(D.DEVICE()))
        self.reset_parameters()
    
    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_feats) + ' -> ' \
               + str(self.out_feats) + ')'

    def reset_parameters(self):
        # stdv = math.sqrt(6.0/ (self.weight.shape[0] + self.weight.shape[1]))
        # self.weight.data.uniform_(-stdv,stdv)
        # self.bias.data.uniform_(-stdv,stdv)

        nn.init.xavier_uniform_(self.weight.data)
        nn.init.xavier_uniform_(self.loop_weight.data)
        return

    def forward(self,input):
        x=input 
        out = torch.matmul(x,self.weight)
        out_loop = torch.matmul(x,self.loop_weight)
        out = dot(self.adj_mat,out,True) + out_loop
        out = out + self.bias
        return out  
    
    def __repr__(self):
        return '{} ({} -> {})'.format(self.__class__.__name__,
                                    self.in_feats,
                                    self.out_feats)



class GraphProjection(nn.Module):
    """Graph Projection layer, which pool 2D features to mesh

    The layer projects a vertex of the mesh to the 2D image and use 
    bilinear interpolation to get the corresponding feature.
    """

    def __init__(self,mesh_pos=[0., 0., -0.8],camera_f=[248., 248.],camera_c=[111.5, 111.5],bound=0):
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
        if self.bound != 0:
            self.threshold = Threshold(bound, bound)

    def bound_val(self, x):
        """
        given x, return min(threshold, x), in case threshold is not None
        """
        if self.bound < 0:
            return -self.threshold(-x)
        elif self.bound > 0:
            return self.threshold(x)
        return x
    
    # Returns width and height of imge feature map
    @staticmethod
    def image_feature_shape(img):
        return np.array([img.size(-1), img.size(-2)])
    
    def project(self, img_shape, img_feature,sample_points,tensorflow_mode=False):
        """
        :param img_shape: raw image shape
        :param img_feature: [batch_size x channel x h x w] batch of image features
        :param sample_points: [batch_size x num_points x 2], in range [-1, 1]
        :return: [batch_size x num_points x feat_dim]
        """
        if tensorflow_mode:
            feature_shape = self.image_feature_shape(img_feature)

            x_shape = img_shape[0]/feature_shape[0]
            y_shape = img_shape[1]/feature_shape[1]
            points_x = sample_points[:,:,0] / x_shape
            points_y = sample_points[:,:,1] / y_shape
            
            img_feat_batch_size = img_feature.shape[0]
            output=[]
            for i in range(img_feat_batch_size):
                output.append(self.project_tensorflow(points_x[i],points_y[i],feature_shape,img_feature[i]))
        else:
            sample_points_ = sample_points.unsqueeze(1)
            output = F.grid_sample(img_feature,sample_points_,align_corners=False)
            output = torch.transpose(output.squeeze(2),1,2)

        return output
    
    def project_tensorflow(self,x,y,feat_shape,img_feat):
        """
        :param x: [n_verts] mesh vertex values along x axis
        :param y: [n_verts] mesh vertex values along y axis
        :param feat_shape: [h,w], height and width of feature map 
        :param img_feat: [c,h,w], feature maps
        :return: output, mesh features concatenated with projected image features
        """
        
        x = torch.clamp(x,min=0, max = feat_shape[0] - 1)
        y = torch.clamp(y,min=0, max = feat_shape[1] - 1)

        x_min, x_max = torch.floor(x).long(), torch.ceil(x).long()
        y_min, y_max = torch.floor(y).long(), torch.ceil(y).long()

        # Get topright,topleft,botright,botleft nearest image features of img_feat
        Q11 = img_feat[:, x_min, y_min].clone()
        Q12 = img_feat[:, x_min, y_max].clone()
        Q21 = img_feat[:, x_max, y_min].clone()
        Q22 = img_feat[:, x_max, y_max].clone()

        x, y = x.long(), y.long()

        weights = torch.mul(x_max - x, y_max - y)
        Q11 = torch.mul(weights.float().view(-1, 1), torch.transpose(Q11, 0, 1))

        weights = torch.mul(x_max - x, y - y_min)
        Q12 = torch.mul(weights.float().view(-1, 1), torch.transpose(Q12, 0 ,1))

        weights = torch.mul(x - x_min, y_max - y)
        Q21 = torch.mul(weights.float().view(-1, 1), torch.transpose(Q21, 0, 1))

        weights = torch.mul(x - x_min, y - y_min)
        Q22 = torch.mul(weights.float().view(-1, 1), torch.transpose(Q22, 0, 1))

        output = Q11 + Q21 + Q12 + Q22
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
        positions = shape_verts + torch.tensor(self.mesh_pos, device=shape_verts.device, dtype=torch.float)

        # Tensorflow (original) version: h = 248 * tf.divide(-Y, -Z) + 112
        y_pos = positions[:,:, 1]
        z_pos = self.bound_val(positions[:,:, 2])
        h = self.camera_f[1] * (y_pos / z_pos) + camera_c_offset[1]
        # Tensorflow (original) version: w = 248.0 * tf.divide(X, -Z) + 112.0
        x_pos = positions[:,:, 0]
        w = -self.camera_f[0] * (x_pos / z_pos) + camera_c_offset[0]

        # directly do clamping
        w /= half_res[0]
        h /= half_res[1]

        # clamp to [-1, 1]
        w = torch.clamp(w, min=-1, max=1)
        h = torch.clamp(h, min=-1, max=1)

        feats = [shape_verts]
        for img_feature in img_features:
            yes = torch.stack([w,h],dim=-1)
            feats.append(self.project(img_res,img_feature,yes))
        output = torch.cat(feats,2)
        return output
   
class GraphUnpoolingLayer(nn.Module):
    """Graph Pooling layer, aims to add additional vertices to the graph.
    The middle point of each edges are added, and its feature is simply
    the average of the two edge vertices.
    Three middle points are connected in each triangle.
    """
    def __init__(self, unpool_idx):
        super(GraphUnpoolingLayer,self).__init__()
        self.unpool_idx = unpool_idx

        # input num verts is highest vertex idx
        self.in_n_verts = torch.max(unpool_idx).item()

        len_ = len(unpool_idx)
        self.out_n_verts = self.in_n_verts + len_
        print()
    
    def forward(self,vertices):
        """
        :param vertices: [b,n_verts,n_channels] vertices of mesh
        :return updated_vertices: [b,n_verts,n_channels] mesh with more added vertices
        """
        
        # new feats are the vertices selected at unpool_idx
        new_features = vertices[:,self.unpool_idx].clone()
        new_vertices = 0.5 * new_features.sum(2)

        # Append new vertices to existing list of vertices
        updated_vertices = torch.cat([vertices,new_vertices],1)
        return updated_vertices
    
    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_n_verts) + ' -> ' \
               + str(self.out_n_verts) + ')'


class VGG16_Encoder(nn.Module):

    def __init__(self,n_input_channels,device=D.DEVICE()):
        super(VGG16_Encoder,self).__init__()
        _ = models.vgg16(pretrained=True).eval().to(device)
        self.features = _.features

        self.total_n_feats = 0

        for param in self.features.parameters():
            param.requires_grad = False
    
    def forward(self,x,layers:dict=None):

        extracted_feats = {}
        for name, layer in self.features._modules.items():
            x = layer(x)
            if layers is not None and name in layers:
                extracted_feats[layers[name]]=x
        if layers:
            return extracted_feats
        return x

class VGG16P2M(nn.Module):
    def __init__(self, n_classes_input=3, pretrained=False, pretrained_path=None):
        super(VGG16P2M, self).__init__()

        self.features_dim = 960

        self.conv0_1 = nn.Conv2d(n_classes_input, 16, 3, stride=1, padding=1)
        self.conv0_2 = nn.Conv2d(16, 16, 3, stride=1, padding=1)

        self.conv1_1 = nn.Conv2d(16, 32, 3, stride=2, padding=1)  # 224 -> 112
        self.conv1_2 = nn.Conv2d(32, 32, 3, stride=1, padding=1)
        self.conv1_3 = nn.Conv2d(32, 32, 3, stride=1, padding=1)

        self.conv2_1 = nn.Conv2d(32, 64, 3, stride=2, padding=1)  # 112 -> 56
        self.conv2_2 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.conv2_3 = nn.Conv2d(64, 64, 3, stride=1, padding=1)

        self.conv3_1 = nn.Conv2d(64, 128, 3, stride=2, padding=1)  # 56 -> 28
        self.conv3_2 = nn.Conv2d(128, 128, 3, stride=1, padding=1)
        self.conv3_3 = nn.Conv2d(128, 128, 3, stride=1, padding=1)

        self.conv4_1 = nn.Conv2d(128, 256, 5, stride=2, padding=2)  # 28 -> 14
        self.conv4_2 = nn.Conv2d(256, 256, 3, stride=1, padding=1)
        self.conv4_3 = nn.Conv2d(256, 256, 3, stride=1, padding=1)

        self.conv5_1 = nn.Conv2d(256, 512, 5, stride=2, padding=2)  # 14 -> 7
        self.conv5_2 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
        self.conv5_3 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
        self.conv5_4 = nn.Conv2d(512, 512, 3, stride=1, padding=1)

        if pretrained and pretrained_path is not None :
            state_dict = torch.load(pretrained_path)
            self.load_state_dict(state_dict)
        else:
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, img):
        img = F.relu(self.conv0_1(img))
        img = F.relu(self.conv0_2(img))
        # img0 = torch.squeeze(img) # 224

        img = F.relu(self.conv1_1(img))
        img = F.relu(self.conv1_2(img))
        img = F.relu(self.conv1_3(img))
        # img1 = torch.squeeze(img) # 112

        img = F.relu(self.conv2_1(img))
        img = F.relu(self.conv2_2(img))
        img = F.relu(self.conv2_3(img))
        img2 = img

        img = F.relu(self.conv3_1(img))
        img = F.relu(self.conv3_2(img))
        img = F.relu(self.conv3_3(img))
        img3 = img

        img = F.relu(self.conv4_1(img))
        img = F.relu(self.conv4_2(img))
        img = F.relu(self.conv4_3(img))
        img4 = img

        img = F.relu(self.conv5_1(img))
        img = F.relu(self.conv5_2(img))
        img = F.relu(self.conv5_3(img))
        img = F.relu(self.conv5_4(img))
        img5 = img

        return [img2, img3, img4, img5]

# ##############################################################

class Pix2VoxF_Encoder(nn.Module):
    def __init__(self):
        super(Pix2VoxF_Encoder,self).__init__()

        
        vgg16_bn = models.vgg16_bn(pretrained=True)
        # Don't update params in VGG16
        for param in vgg16_bn.parameters():
            param.requires_grad = False
        self.vgg = nn.Sequential(*list(vgg16_bn.features.children()))[:27]

        self.layer1 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=1),
            nn.BatchNorm2d(512),
            nn.ELU(),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3),
            nn.BatchNorm2d(256),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=4)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3),
            nn.BatchNorm2d(128),
            nn.ELU()
        )
    
    def forward(self,images):
        # print(rendering_images.size())  # torch.Size([batch_size, n_views, img_c, img_h, img_w])
        rendering_images = images.permute(1, 0, 2, 3, 4).contiguous()
        rendering_images = torch.split(rendering_images, 1, dim=0)
        image_features = []

        for img in rendering_images:
            features = self.vgg(img.squeeze(dim=0))
            # print(features.size())    # torch.Size([batch_size, 512, 28, 28])
            features = self.layer1(features)
            # print(features.size())    # torch.Size([batch_size, 512, 28, 28])
            features = self.layer2(features)
            # print(features.size())    # torch.Size([batch_size, 256, 6, 6])
            features = self.layer3(features)
            # print(features.size())    # torch.Size([batch_size, 128, 4, 4])
            image_features.append(features)

        image_features = torch.stack(image_features).permute(1, 0, 2, 3, 4).contiguous()
        # print(image_features.size())  # torch.Size([batch_size, n_views, 128, 4, 4])
        return image_features


class Pix2VoxF_Decoder(nn.Module):
    def __init__(self):
        super(Pix2VoxF_Decoder,self).__init__()

        # Layer Definition
        self.layer1 = nn.Sequential(
            nn.ConvTranspose3d(256, 128, kernel_size=4, stride=2, bias=True, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU()
        )
        self.layer2 = nn.Sequential(
            nn.ConvTranspose3d(128, 64, kernel_size=4, stride=2, bias=True, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU()
        )
        self.layer3 = nn.Sequential(
            nn.ConvTranspose3d(64, 32, kernel_size=4, stride=2, bias=True, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU()
        )
        self.layer4 = nn.Sequential(
            nn.ConvTranspose3d(32, 8, kernel_size=4, stride=2, bias=True, padding=1),
            nn.BatchNorm3d(8),
            nn.ReLU()
        )
        self.layer5 = nn.Sequential(
            nn.ConvTranspose3d(8, 1, kernel_size=1, bias=True),
            nn.Sigmoid()
        )
    

    def forward(self, image_features):
        image_features = image_features.permute(1, 0, 2, 3, 4).contiguous()
        image_features = torch.split(image_features, 1, dim=0)
        gen_voxels = []
        raw_features = []

        for features in image_features:
            gen_voxel = features.view(-1, 256, 2, 2, 2)
            # print(gen_voxel.size())   # torch.Size([batch_size, 256, 2, 2, 2])
            gen_voxel = self.layer1(gen_voxel)
            # print(gen_voxel.size())   # torch.Size([batch_size, 128, 4, 4, 4])
            gen_voxel = self.layer2(gen_voxel)
            # print(gen_voxel.size())   # torch.Size([batch_size, 64, 8, 8, 8])
            gen_voxel = self.layer3(gen_voxel)
            # print(gen_voxel.size())   # torch.Size([batch_size, 32, 16, 16, 16])
            gen_voxel = self.layer4(gen_voxel)
            # print(gen_voxel.size())   # torch.Size([batch_size, 8, 32, 32, 32])
            raw_feature = gen_voxel
            gen_voxel = self.layer5(gen_voxel)
            # print(gen_voxel.size())   # torch.Size([batch_size, 1, 32, 32, 32])
            raw_feature = torch.cat((raw_feature, gen_voxel), dim=1)
            # print(raw_feature.size()) # torch.Size([batch_size, 9, 32, 32, 32])

            gen_voxels.append(torch.squeeze(gen_voxel, dim=1))
            raw_features.append(raw_feature)

        gen_voxels = torch.stack(gen_voxels).permute(1, 0, 2, 3, 4).contiguous()
        raw_features = torch.stack(raw_features).permute(1, 0, 2, 3, 4, 5).contiguous()
        # print(gen_voxels.size())        # torch.Size([batch_size, n_views, 32, 32, 32])
        # print(raw_features.size())      # torch.Size([batch_size, n_views, 9, 32, 32, 32])
        return raw_features, gen_voxels


# ##############################################################
# class GResBlock(nn.Module):

#     def __init__(self, in_dim, hidden_dim, adjs, use_cuda):
#         super(GResBlock, self).__init__()

#         self.conv1 = GraphConvolution(in_features = in_dim, out_features = hidden_dim, adjs = adjs, use_cuda = use_cuda)
#         self.conv2 = GraphConvolution(in_features = in_dim, out_features = hidden_dim, adjs = adjs, use_cuda = use_cuda) 


#     def forward(self, input):
        
#         x = self.conv1(input)
#         x = self.conv2(x)

#         return (input + x) * 0.5


# class GBottleneck(nn.Module):

#     def __init__(self, block_num, in_dim, hidden_dim, out_dim, adjs, use_cuda):
#         super(GBottleneck, self).__init__()

#         blocks = [GResBlock(in_dim = hidden_dim, hidden_dim = hidden_dim, adjs = adjs, use_cuda = use_cuda)]

#         for _ in range(block_num - 1):
#             blocks.append(GResBlock(in_dim = hidden_dim, hidden_dim = hidden_dim, adjs = adjs, use_cuda = use_cuda))

#         self.blocks = nn.Sequential(*blocks)
#         self.conv1 = GraphConvolution(in_features = in_dim, out_features = hidden_dim, adjs = adjs, use_cuda = use_cuda)
#         self.conv2 = GraphConvolution(in_features = hidden_dim, out_features = out_dim, adjs = adjs, use_cuda = use_cuda)

        
#     def forward(self, input):

#         x = self.conv1(input)
#         x_cat = self.blocks(x)
#         x_out = self.conv2(x_cat)

#         return x_out, x_cat

# class P2M_Model(nn.Module):
#     """
#     Implement the joint model for Pixel2mesh
#     """

#     def __init__(self, features_dim, hidden_dim, coord_dim, pool_idx, supports, use_cuda):

#         super(P2M_Model, self).__init__()
#         self.img_size = 224

#         self.features_dim = features_dim
#         self.hidden_dim = hidden_dim
#         self.coord_dim = coord_dim
#         self.pool_idx = pool_idx
#         self.supports = supports
#         self.use_cuda = use_cuda

#         self.build()


#     def build(self):

#         self.nn_encoder = self.build_encoder()
#         self.nn_decoder = self.build_decoder()

#         self.GCN_0 = GBottleneck(6, self.features_dim, self.hidden_dim, self.coord_dim, self.supports[0], self.use_cuda)
#         self.GCN_1 = GBottleneck(6, self.features_dim + self.hidden_dim, self.hidden_dim, self.coord_dim, self.supports[1], self.use_cuda)
#         self.GCN_2 = GBottleneck(6, self.features_dim + self.hidden_dim, self.hidden_dim, self.hidden_dim, self.supports[2], self.use_cuda)

#         self.GPL_1 = GraphPooling(self.pool_idx[0])
#         self.GPL_2 = GraphPooling(self.pool_idx[1])

#         # GPR projects image features onto mesh
#         self.GPR_0 = GraphProjection() 
#         self.GPR_1 = GraphProjection()
#         self.GPR_2 = GraphProjection()

#         self.GConv = GraphConvolution(in_features = self.hidden_dim, out_features = self.coord_dim, adjs = self.supports[2], use_cuda = self.use_cuda)

#         self.GPL_12 = GraphPooling(self.pool_idx[0])
#         self.GPL_22 = GraphPooling(self.pool_idx[1])


#     def forward(self, img, input):

#         img_feats = self.nn_encoder(img)

#         # GCN Block 1
#         x = self.GPR_0(img_feats, input)
#         x1, x_cat = self.GCN_0(x)
#         x1_2 = self.GPL_12(x1)

#         # GCN Block 2
#         x = self.GPR_1(img_feats, x1)
#         x = torch.cat([x, x_cat], 1)
#         x = self.GPL_1(x)
#         x2, x_cat = self.GCN_1(x)
#         x2_2 = self.GPL_22(x2)
        
#         # GCN Block 3
#         x = self.GPR_2(img_feats, x2)
#         x = torch.cat([x, x_cat], 1)
#         x = self.GPL_2(x)
#         x, _ = self.GCN_2(x)

#         x3 = self.GConv(x)

#         new_img = self.nn_decoder(img_feats)

#         return [x1, x2, x3], [input, x1_2, x2_2], new_img


#     def build_encoder(self):
#         # VGG16 at first, then try resnet
#         # Can load params from model zoo
#         net = VGG16_Encoder(n_classes_input = 3)
#         return net

#     def build_decoder(self):
#         net = VGG16_Decoder()
#         return net

# # Graph Convolutional Network
# # ##############################################################
# def torch_sparse_tensor(indice, value, size, use_cuda):

#     coo = coo_matrix((value, (indice[:, 0], indice[:, 1])), shape = size)
#     values = coo.data
#     indices = np.vstack((coo.row, coo.col))

#     i = torch.LongTensor(indices)
#     v = torch.FloatTensor(values)
#     shape = coo.shape

#     if use_cuda:
#         return torch.sparse.FloatTensor(i, v, shape).cuda()
#     else:
#         return torch.sparse.FloatTensor(i, v, shape)


# def dot(x, y, sparse = False):
#     """Wrapper for torch.matmul (sparse vs dense)."""
#     if sparse:
#         res = x.mm(y)
#     else:
#         res = torch.matmul(x, y)
#     return res


# class GraphConvolution(nn.Module):
#     """Simple GCN layer
#     Similar to https://arxiv.org/abs/1609.02907
#     """
#     def __init__(self, in_features, out_features, adjs, bias=True, use_cuda = True):
#         super(GraphConvolution, self).__init__()
#         self.in_features = in_features
#         self.out_features = out_features

#         adj0 = torch_sparse_tensor(*adjs[0], use_cuda)
#         adj1 = torch_sparse_tensor(*adjs[1], use_cuda)
#         self.adjs = [adj0, adj1]

#         self.weight_1 = nn.Parameter(torch.FloatTensor(in_features, out_features))
#         self.weight_2 = nn.Parameter(torch.FloatTensor(in_features, out_features))

#         if bias:
#             self.bias = nn.Parameter(torch.FloatTensor(out_features))
#         else:
#             self.register_parameter('bias', None)
#         self.reset_parameters()

#     def reset_parameters(self):
#         stdv = 1. / math.sqrt(self.weight_1.size(1))
#         self.weight_1.data.uniform_(-stdv, stdv)
#         self.weight_2.data.uniform_(-stdv, stdv)
#         if self.bias is not None:
#             self.bias.data.uniform_(-stdv, stdv)

#     def forward(self, input):
#         support_1 = torch.matmul(input, self.weight_1)
#         support_2 = torch.matmul(input, self.weight_2)
#         #output = torch.spmm(adj, support)
#         output1 = dot(self.adjs[0], support_1, True)
#         output2 = dot(self.adjs[1], support_2, True)
#         output = output1 + output2
#         if self.bias is not None:
#             return output + self.bias
#         else:
#             return output

#     def __repr__(self):
#         return self.__class__.__name__ + ' (' \
#             + str(self.in_features) + ' -> ' \
#             + str(self.out_features) + ')'


# class GraphPooling(nn.Module):
#     """Graph Pooling layer, aims to add additional vertices to the graph.

#     The middle point of each edges are added, and its feature is simply
#     the average of the two edge vertices.
#     Three middle points are connected in each triangle.
#     """

#     def __init__(self, pool_idx):
#         super(GraphPooling, self).__init__() 
#         self.pool_idx = pool_idx
#         # save dim info
#         self.in_num = np.max(pool_idx)
#         self.out_num = self.in_num + len(pool_idx)

#     def forward(self, input):

#         new_features = input[self.pool_idx].clone()
#         new_vertices = 0.5 * new_features.sum(1)
#         output = torch.cat((input, new_vertices), 0)

#         return output

#     def __repr__(self):
#         return self.__class__.__name__ + ' (' \
#                + str(self.in_num) + ' -> ' \
#                + str(self.out_num) + ')'




# ##############################################################