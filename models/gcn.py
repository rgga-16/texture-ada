
import torch as th
import torch.nn as nn
import torch.nn.functional as F 

import numpy as np



'''
Parameters:
in_feats - No. of input features 
out_feats - No. of output features
adj_mat - Adjustment matrix to adjust the shape vertices
bias - Bias value to add to the output 
'''
class GraphConvolutionLayer(nn.Module):
    def __init__(self,in_feats,out_feats,adj_mat, bias=True):
        super(GraphConvolutionLayer,self).__init__()
        self.in_feats = in_feats
        self.out_feats = out_feats
        # Set adj_mat as a Parameter to be able to adjust it in grad descent?
        self.adj_mat = nn.Parameter(adj_mat,requires_grad=False)

        # Weights
        self.weight = nn.Parameter(torch.zeros((in_feats,out_feats),dtype=th.float))
        self.loop_weight = nn.Parameter(torch.zeros((in_feats,out_feats),dtype=th.float))

        # Bias
        if bias:
            self.bias = nn.Parameter(torch.zeros((out_feats,),dtype=th.float))
        else:
            self.register_parameter('bias',None)
    
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight.data)
        nn.init.xavier_uniform_(self.loop_weight.data)
    
    def forward(self,inputs):
        support = th.matmul(inputs,self.weight)
        support_loop = th.matmul(inputs,self.loop_weight)
        output = th.matmul(self.adj_mat,support) + support_loop

        if self.bias is not None:
            output = output + self.bias

        return output


class GraphResBlock(nn.Module):  
    def __init__(self, in_feats, hidden_feats,adj_mat):
        super(GraphResBlock,self).__init__()
        self.conv1 = GraphConvolutionLayer(in_feats,hidden_feats,adj_mat)
        self.conv2 = GraphConvolutionLayer(hidden_feats,in_feats,adj_mat)
        self.activation = F.relu

    def forward(self, inputs):

        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.activation(x)

        return (inputs + x) * 0.5

'''
Deformation block for returning the new vertex locations
'''
class GraphBottleNeck(nn.Module):
    def __init__(self,num_blocks,in_feats,hidden_feats,out_feats,adj_mat):
        super(GraphBottleNeck,self).__init__()

        self.conv1 = GraphConvolutionLayer(in_feats,hidden_feats,adj_mat)

        resblocks = [GraphResBlock(in_feats,hidden_feats,adj_mat) for i in range(num_blocks) ]
        self.resblocks = nn.Sequential(*resblocks)

        self.conv2 = GraphConvolutionLayer(hidden_feats,out_feats,adj_mat)
        self.activation = F.relu

    def forward(self,inputs):
        x = self.conv1(inputs)
        x = self.activation(x)
        hidden = self.resblocks(x)
        output = self.conv2(hidden)

        return output,hidden

"""
Adds vertices to the graph.

1) Add midpoint to each edge (average of the two edge vertices)
2) 3 midpoints are connected in each triangle
"""
class GraphUnpoolingLayer(nn.Module):
    def __init__(self,unpool_idx):
        super(GraphUnpoolingLayer,self).__init__()

        self.input_num = th.max(unpool_idx).item()
        self.output_num = self.input_num + len(unpool_idx)
    
    def forward(self,inputs):

        new_feats = inputs[:,self.unpool_idx].clone()
        new_verts = .5 * new_feats.sum(2)
        output = th.cat([inputs,new_verts],1)

        return output

"""
Pools 2D features into the mesh
The layer projects a vertex of the mesh to the 2D image and use
bi-linear interpolation to get the corresponding feature.
"""
class GraphProjectionLayer(nn.Module):

    def __init__(self, mesh_pos, camera_f, camera_c, bound=0):
        super(GraphProjectionLayer,self).__init__()
        self.mesh_pos, self.camera_f, self.camera_c = mesh_pos, camera_f, camera_c
        self.bound = 0

    
    @staticmethod
    def image_feature_shape(img):
        return np.array([img.size(-1), img.size(-2)])
    
    def bound_val(self, x):
        """
        given x, return min(threshold, x), in case threshold is not None
        """
        if self.bound < 0:
            return -self.threshold(-x)
        elif self.bound > 0:
            return self.threshold(x)
        return x
    
    def project(self,img_shape, img_feat, sample_points):
        """
        :param img_shape: raw image shape
        :param img_feat: [batch_size x channel x h x w]
        :param sample_points: [batch_size x num_points x 2], in range [-1, 1]
        :return: [batch_size x num_points x feat_dim]
        """
        output = F.grid_sample(img_feat, sample_points.unsqueeze(1))
        output = th.transpose(output.squeeze(2),1,2)
        return output

    def forward(self, resolution, img_features, inputs):
        half_res = (resolution-1)/2

        camera_c_offset = np.array(self.camera_c) - half_resolution

        positions = inputs + th.tensor(self.mesh_pos, device=inputs.device, dtype=th.float)
        w = -self.camera_f[0] * (positions[:, :, 0] / self.bound_val(positions[:, :, 2])) + camera_c_offset[0]
        h = self.camera_f[1] * (positions[:, :, 1] / self.bound_val(positions[:, :, 2])) + camera_c_offset[1]

        w /= half_res[0]
        h /= half_res[1]

        # clamp to [-1, 1]
        w = th.clamp(w, min=-1, max=1)
        h = th.clamp(h, min=-1, max=1)

        feats = [inputs]
        for img_feature in img_features:
            feats.append(self.project(resolution, img_feature, th.stack([w, h], dim=-1)))

        output = th.cat(feats, 2)

        return output




class GraphConvolutionalNetwork(nn.Module):
    def __init__(self, ellipsoid, features_dim, hidden_dim, coord_dim, last_hidden_dim, camera_f, camera_c, mesh_pos):
        super(GraphConvolutionalNetwork,self).__init__()

        self.features_dim = features_dim
        self.hidden_dim = hidden_dim
        self.coord_dim = coord_dim
        self.last_hidden_dim = last_hidden_dim

        self.deformation_blocks = nn.ModuleList([
            GraphBottleNeck(6,self.features_dim,self.hidden_dim,self.coord_dim,ellipsoid.adj_mat[0]),
            GraphBottleNeck(6,self.features_dim + self.hidden_dim, self.hidden_dim, self.coord_dim, ellipsoid.adj_mat[1]),
            GraphBottleNeck(6, self.features_dim + self.hidden_dim, self.hidden_dim, self.last_hidden_dim,elllipsoid.adj_mat[2])

        ])

        self.unpooling = nn.ModuleList([
            GraphUnpoolingLayer(ellipsoid.unpool_idx[0]),
            GraphUnpoolingLayer(ellipsoid.unpool_idx[1])
        ])

        self.projection = GraphProjectionLayer(mesh_pos,camera_f)


    def forward(self,img):

        # Batchsize
        batch_size = img.size(0)

        # First deform block
        x = self.proj







