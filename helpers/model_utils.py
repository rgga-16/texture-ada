
from torchvision import transforms
import torch

from defaults import DEFAULTS as D

import kaolin as kal
import open3d as o3d
import numpy as np, random, math

import os


class NormalizePointcloud(object):
    def __call__(self,pointcloud):
        assert pointcloud.dim()==3 and pointcloud.shape[2]==3
        pointcloud = pointcloud - torch.mean(pointcloud,dim=1)
        pointcloud /= torch.max(torch.linalg.norm(pointcloud,dim=2))
        return pointcloud
    
class NormalizePointcloudMinMax(object):
    def __call__(self, pointcloud):
        assert pointcloud.dim()==3 and pointcloud.shape[2]==3
        min = torch.min(pointcloud,dim=1,keepdim=True)[0]
        max = torch.max(pointcloud,dim=1,keepdim=True)[0]
        pointcloud = pointcloud-min
        pointcloud = pointcloud / max 
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
                [0,0,1],
            ],device=D.DEVICE())

        pointcloud = torch.matmul(pointcloud,rot_matrix)
        return pointcloud

class Y_Rotate(object):
    def __init__(self, theta):
        self.theta=theta
    
    def __call__(self,pointcloud):
        rot_matrix = torch.tensor(
            [
                [math.cos(self.theta),0,math.sin(self.theta)],
                [0,1,0],
                [-math.sin(self.theta),0,math.cos(self.theta)], 
            ],device=D.DEVICE())

        pointcloud = torch.matmul(pointcloud,rot_matrix)
        return pointcloud

class AddNoise(object):
    def __call__(self,pointcloud):
        noise = torch.normal(mean=0.0,std=0.02,size=pointcloud.shape)
        noise = noise.to(D.DEVICE())
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

    vertices=vertices.to(D.DEVICE())
    faces = faces.to(D.DEVICE())

    vertices = normalize_vertices(vertices)
    adj = kal.ops.mesh.adjacency_matrix(vertices.shape[0],faces).clone()
	
    return vertices,faces,adj

def mesh_to_pointcloud(vertices,faces,n_points=3000,device=D.DEVICE()):
    transformer = transforms.Compose([
        SamplePoints(n_points),
        NormalizePointcloud(),
        Y_Rotate(67.5)
        # AddNoise()
    ])

    pointcloud = transformer((vertices,faces))

    return pointcloud.to(device)

'''
Reconstructs mesh from pointcloud using Poisson surface reconstruction method (Kahzdan et al, 2006)
Reference: http://hhoppe.com/poissonrecon.pdf 
'''
def pointcloud_to_mesh_poisson(pointcloud,depth=5):
    pointcloud.normals = o3d.utility.Vector3dVector(np.zeros((1,3)))
    pointcloud.estimate_normals()
    mesh,_ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pointcloud,depth=depth,width=0,scale=1.1,linear_fit=True)
    mesh = mesh.crop(pointcloud.get_axis_aligned_bounding_box())
    mesh.compute_vertex_normals()
    return mesh

def pointcloud_to_mesh_ballpivot(pointcloud, radii = [0.005, 0.01, 0.02, 0.04]):
    pointcloud.normals = o3d.utility.Vector3dVector(np.zeros((1,3)))
    pointcloud.estimate_normals()
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pointcloud,o3d.utility.DoubleVector(radii))
    mesh = mesh.crop(pointcloud.get_axis_aligned_bounding_box())
    mesh.compute_vertex_normals()
    return mesh 


def pointcloud_kaolin_to_open3d(pointcloud):
    pointcloud_o3d = o3d.geometry.PointCloud()
    pointcloud_o3d.points = o3d.utility.Vector3dVector(pointcloud.squeeze().detach().cpu().numpy())
    return pointcloud_o3d