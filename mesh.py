import sys
import torch 

import pytorch3d.utils as u
from pytorch3d.ops import sample_points_from_meshes
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D

def create_sphere_mesh(level=5,device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')):
    sphere = u.ico_sphere(level=level,device=device)

    return sphere

def plot_pointcloud(mesh, title=""):
    # Sample points uniformly from the surface of the mesh.
    points = sample_points_from_meshes(mesh, 5000)
    x, y, z = points.clone().detach().cpu().squeeze().unbind(1)    
    fig = plt.figure(figsize=(5, 5))
    ax = Axes3D(fig)
    ax.scatter3D(x, z, -y)
    ax.set_xlabel('x')
    ax.set_ylabel('z')
    ax.set_zlabel('y')
    ax.set_title(title)
    ax.view_init(190, 30)
    plt.show(block=False)

if __name__ == "__main__":
    s = create_sphere_mesh()
    plot_pointcloud(s,'Sphere')
    
