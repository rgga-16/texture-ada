import torch 
import torchvision 

import pytorch3d
from defaults import DEFAULTS as D 

from pathlib import Path

from pytorch3d.renderer.mesh import TexturesAtlas, TexturesUV, TexturesVertex

device = D.DEVICE()

obj_dir = './inputs/cow_mesh'
obj_file = obj_dir / 'cow.obj'

# Load mesh and texture
verts, faces, aux = load_obj(
    obj_file, 
    device=device, 
    load_textures=True,
    create_texture_atlas=True,
    texture_atlas_size=8,
    texture_wrap=None,
)

atlas = aux.texture_atlas 

mesh= Meshes(
    verts=[verts],
    faces=[faces.verts_idx],
    textures=TexturesAtlas(atlas=[atlas])
)

# Init rasterizer settings
R, T = look_at_view_transform(2.7, 0, 0)
cameras = FoVPerspectiveCameras(device=device, R=R, T=T)

raster_settings = RasterizationSettings(
    image_size=512,
    blur_radius=0.0,
    faces_per_pixel=1,
    cull_backfaces=True,
)

# Init shader settings
materials = Materials(device=device, specular_color=((0, 0, 0),), shininess=0.0)
lights = PointLights(device=device)

# Place light behind the cow in world space. The front of
# the cow is facing the -z direction.
lights.location = torch.tensor([0.0, 0.0, 2.0], device=device)[None]

# The HardPhongShader can be used directly with atlas textures.
rasterizer = MeshRasterizer(cameras=cameras, raster_settings=raster_settings)
renderer = MeshRenderer(
    rasterizer=rasterizer,
    shader=HardPhongShader(lights=lights, cameras=cameras, materials=materials),
)

images = renderer(mesh)
rgb = images[0, ..., :3].squeeze()

print(rgb.shape)