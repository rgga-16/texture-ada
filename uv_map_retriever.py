
import os
import pathlib as p

from blender import BlenderRenderer

from PIL import Image

import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(
        description='UV Map Retrieval. Uses Blender API to obtain the UV maps of meshes')
    parser.add_argument('--mode',type=str,help='Retrieve single UV map (SINGLE) or multiple UV maps (MULTI)',
                        default='MULTI')
    parser.add_argument('--mesh', type=str,help='Path to the mesh to obtain UV map from',
                        default=None)     
    parser.add_argument('--mesh_dir', type=str,
                        help='Path to the mesh directory',
                        default="inputs/shape_samples/lounge_sofa")                 
    parser.add_argument('--uv', type=str,help='Output path to save the obtained UV map',
                        default=None)
    parser.add_argument('--uv_dir', type=str,help='Path of the output directory to save the UVs',
                        default="inputs/uv_maps/lounge_sofa/unwrap")
    return parser.parse_args()


if __name__ == '__main__':
    renderer = BlenderRenderer()
    args = parse_arguments()

    assert args.mode in ['SINGLE', 'MULTI']

    assert args.mesh is not None or args.mesh_dir is not None 

    assert args.uv is not None or args.uv_dir is not None 

    if args.mode=='SINGLE':
        obj = renderer.load_object(args.mesh)
        renderer.save_uv_map(obj,save_file=args.uv)
    else:
        for mesh_file in os.listdir(args.mesh_dir):
            ext = os.path.splitext(mesh_file)[-1].lower()
            if ext in ['.obj']:
                mesh_path = os.path.join(args.mesh_dir,mesh_file)
                obj = renderer.load_object(mesh_path)
                uv_path = str(p.Path.cwd() / args.uv_dir / '{}_uv.png'.format(mesh_file[:-4]))
                renderer.save_uv_map(obj,save_file=uv_path)


    # # armchair sofa
    # meshes_dir = args.mesh_dir
    # uv_maps_dir = args.uv_dir

    # # Add all objects and their textures into scene
    # for mesh_file,texture_file in mesh_texture_file_pairs.items():
    #     mesh_path = os.path.join(meshes_dir,mesh_file)
    #     obj = renderer.load_object(mesh_path)
    #     uv_path = str(p.Path.cwd() / uv_maps_dir / 'uv_map_{}.png'.format(mesh_file[:-4]))
    #     renderer.save_uv_map(obj,save_file=uv_path)
        # renderer.apply_texture(obj,texture_path)
