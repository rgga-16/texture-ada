import struct
import torch
import torchvision
import numpy as np
import losses
import os 
from args import args
from helpers import utils
from models import structure_transfer_models
from models import texture_transfer_models
from defaults import DEFAULTS as D
import pickle

torch.manual_seed(0)

import copy 
import datetime 
import kaolin as kal
import random

from scipy.sparse import coo_matrix

def torch_sparse_tensor(indices, value, size):
    row = indices[:,0]
    col = indices[:,1]
    coo = coo_matrix((value, (row, col)), shape=size)
    values = coo.data
    indices = np.vstack((coo.row, coo.col))

    i = torch.tensor(indices, dtype=torch.long)
    v = torch.tensor(values, dtype=torch.float)
    shape = coo.shape

    return torch.sparse.FloatTensor(i, v, shape).to(D.DEVICE())

def normalize_vertices(vertices):
    """
    Normalizes vertices to fit an [-1...1] bounding box,
    common during training, but not necessary for visualization.
    """
    result = vertices - torch.mean(vertices, dim=0).unsqueeze(0)
    span = torch.max(result, dim=0).values - torch.min(result, dim=0).values
    return result / torch.max(span)


def setup_meshes(filename,device=D.DEVICE()):
    res = kal.io.obj.import_mesh(filename)
    vertices = res.vertices.to(device)
    faces = res.faces.long().to(device)

    vertices = normalize_vertices(vertices)
    adj = kal.ops.mesh.adjacency_matrix(vertices.shape[0],faces).clone().to(device)
	
    return vertices,faces,adj

def load_dat(filename):
    with open(filename,"rb") as fp:
        fp_info = pickle.load(fp,encoding='latin1')

        print()
    
    return fp_info

# #####################################################

class Ellipsoid(object):

    def __init__(self,filepath,mesh_pos= [0., 0., -0.8]):
        fp_info = load_dat(filepath)

        # verts = (n_verts,3)
        self.verts = torch.tensor(fp_info[0]) - torch.tensor(mesh_pos,dtype=torch.float)
        self.verts = self.verts.to(D.DEVICE())

        # edges & faces & lap_idx
        # edge = (num_edges,2)
        # faces = (num_faces,4)
        # laplace_idx = (num_pts,10)
        self.edges, self.laplace_idx = [], []
        for i in range(3):
            self.edges.append(torch.tensor(fp_info[1 + i][1][0], dtype=torch.long).to(D.DEVICE()))
            self.laplace_idx.append(torch.tensor(fp_info[7][i], dtype=torch.long).to(D.DEVICE()))
        
        # unpool index
        # un_pool = (num_pool_edges 2)
        # unpool_idx[0] = (462,2), 
        # unpool_idx[1] = (1848,2)
        self.unpool_idx = [torch.tensor(fp_info[4][i], dtype=torch.long).to(D.DEVICE()) for i in range(2)]

        # loops and adjacent edges (adjacency matrix)
        self.adj_mat = []
        for i in range(1, 4):
            # Returned as 3 lists where:
            # 0: np.array, 2D, pos
            # 1: np.array, 1D, vals
            # 2: tuple - shape, n * n
            adj_mat_init = fp_info[i][1]
            # Converts to a sparse tensor
            adj_mat_ = torch_sparse_tensor(*adj_mat_init)
            self.adj_mat.append(adj_mat_)
        

        ellipsoid_dir = os.path.dirname(filepath)
        self.faces = []
        self.obj_fmt_faces = []
        # faces: f * 3, original ellipsoid, and two after deformations
        for i in range(1, 4):
            face_file = os.path.join(ellipsoid_dir, "face%d.obj" % i)
            faces_txt = np.loadtxt(face_file, dtype='|S32')
            self.obj_fmt_faces.append(faces_txt)
            faces_ = torch.tensor(faces_txt[:, 1:].astype(np.int) - 1).to(D.DEVICE())
            self.faces.append(faces_)

filename = './inputs/ellipsoid/info_ellipsoid.dat'

ellipsoid = Ellipsoid(filename)

print()

tensor = utils.image_to_tensor(utils.load_image('./inputs/style_images/tiled/chair-3_tiled.png'),image_size=224)
tensor = tensor[:3,...].unsqueeze(0)
b,c,h,w = tensor.shape

encoder = structure_models.VGG16_Encoder(3)

image_feat_vgg16_layers = {
    '4':'maxpool_1', # 1,64,112,112 feat maps
    '9':'maxpool_2', # 1,128,56,56 feat maps
    '16':'maxpool_3', # 1,256,28,28 feat maps
    '23':'maxpool_4', # 1,512,14,14 feat maps
    # '30':'maxpool_5', # 1,512,7,7 feat maps
}

image_feats = encoder(tensor,layers=image_feat_vgg16_layers)
image_feats = list(image_feats.values())
num_image_feats = np.sum([feats.shape[1] for feats in image_feats])

# encoder2 = structure_models.VGG16P2M(3).to(D.DEVICE()).eval()

# 1,64,56,56 feat maps
# 1,128,28,28 feat maps
# 1,256,14,14 feat maps
# 1,512,7,7 feat maps
# image_feats2 = encoder2(tensor)
# num_image_feats_2 = np.sum([feats.shape[1] for feats in image_feats2])
# print()

p2m = structure_models.Pixel2MeshModel(init_shape=ellipsoid)

output_dict = p2m(tensor)
print()

# #####################################################################

# mesh_file = './inputs/ellipsoid/ellipsoid.obj'




# vertices,faces,adj = setup_meshes(mesh_file)
# vertices = torch.nn.Parameter(vertices,requires_grad=False)
# init_verts = vertices.data.unsqueeze(0).expand(b,-1,-1)

# num_epochs=10


# graph_projection = structure_models.GraphProjection()


# concat_verts = graph_projection(np.array([h,w]),image_feats,init_verts)
# print()



# #####################################################################




# def covariance_matrix(tensor):
#     b,c,w,h = tensor.shape
#     feats = tensor.view(b,c,h*w)
#     mean=torch.mean(feats,dim=2,keepdim=True)
#     thing=feats-mean
#     thing = thing.squeeze()
#     print(thing)
#     covariance = torch.mm(thing,thing.t())
#     return covariance / (h*w)


# toy = torch.randn(1,3,2,2)
# print(toy)

# print(covariance_matrix(toy))

# start = datetime.datetime.today().strftime('%m-%d-%y %H-%M-%S')
# date = copy.deepcopy(datetime.datetime.today().strftime('%m-%d-%y %H-%M-%S'))

# output_folder = os.path.join(args.output_dir,"[{}]".format(start))
# try:
#     os.mkdir(output_folder)
# except FileExistsError:
#     pass

# print(start)
# print(date)

