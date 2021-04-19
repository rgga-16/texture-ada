
from torchvision import transforms
import torch

from defaults import DEFAULTS as D

import kaolin as kal
import numpy as np, random, math

import os
from args import args


class NormalizePointcloud(object):
    def __call__(self,pointcloud):
        assert pointcloud.dim()==3 and pointcloud.shape[2]==3
        pointcloud = pointcloud - torch.mean(pointcloud,dim=1)
        pointcloud /= torch.max(torch.linalg.norm(pointcloud,dim=2))
        return pointcloud

class SamplePoints(object):
    def __init__(self,n_points):
        assert isinstance(n_points,int)
        self.n_points=n_points
        
    def __call__(self, mesh):
        vertices,faces=mesh
        if(vertices.dim()==2):
            vertices = vertices.unsqueeze(0)

        pointcloud, _ = kal.ops.mesh.sample_points(vertices,faces,num_samples=self.n_points)
        return pointcloud

class Rotate(object):
    def __call__(self,pointcloud, rotmat):
        return torch.matmul(pointcloud,rotmat)
        

class RandomRotate(object):
    def __call__(self,pointcloud):
        theta = random.random() * 2.0 * math.pi
        rot_matrix = torch.tensor(
            [
                [math.cos(theta),-math.sin(theta),0],
                [math.sin(theta),math.cos(theta),0], 
                [0,0,1]
            ])

        pointcloud = torch.matmul(pointcloud,rot_matrix)
        return pointcloud

class AddNoise(object):
    def __call__(self,pointcloud):
        noise = torch.normal(mean=0.0,std=0.02,size=pointcloud.shape)
        pointcloud = pointcloud+noise 
        return pointcloud

def normalize_vertices(vertices):
    """
    Normalizes vertices to fit an [-1...1] bounding box,
    common during training, but not necessary for visualization.
    """
    result = vertices - torch.mean(vertices, dim=0).unsqueeze(0)
    span = torch.max(result, dim=0).values - torch.min(result, dim=0).values
    return result / torch.max(span)

def load_mesh(filename):
    res = kal.io.obj.import_mesh(filename)
    vertices = res.vertices
    faces = res.faces.long()

    vertices = normalize_vertices(vertices)
    adj = kal.ops.mesh.adjacency_matrix(vertices.shape[0],faces).clone()
	
    return vertices,faces,adj

def mesh_to_pointcloud(vertices,faces,device=D.DEVICE()):
    transformer = transforms.Compose([
        SamplePoints(3000),
        NormalizePointcloud(),
        RandomRotate(),
        AddNoise()
    ])

    pointcloud = transformer((vertices,faces))

    return pointcloud.to(device)