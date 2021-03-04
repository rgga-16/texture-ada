
import math
import os
import pathlib as p

from blender import BlenderRenderer

from PIL import Image

def render_gif():
    return


if __name__ == '__main__':
    renderer = BlenderRenderer()

    # armchair sofa
    meshes_dir = './inputs/shape_samples/armchair sofa'
    textures_dir = 'outputs/output_images/Pyramid2D_with_instnorm/armchair sofa/[2-23-21 20-00]'
    uv_maps_dir = 'inputs/uv_maps/armchair sofa/unwrap'
    mesh_texture_file_pairs = {
        'backseat.obj':'uv_map_backseat_chair-2_masked.png',
        'base.obj':'uv_map_base_chair-1_masked.png',
        'left_arm.obj':'uv_map_left_arm_chair-3_masked.png',
        'left_foot.obj':'uv_map_left_foot_chair-4_masked.png',
        'right_arm.obj':'uv_map_right_arm_chair-3_masked.png',
        'right_foot.obj':'uv_map_right_foot_chair-4_masked.png',
        'seat.obj':'uv_map_seat_chair-6_masked.png',
    }

    # # office chair
    # meshes_dir = './inputs/shape_samples/office chair'
    # textures_dir = 'outputs/output_images/Pyramid2D_with_instnorm/office chair/[3-2-21 7-00]'
    # uv_maps_dir = 'inputs/uv_maps/office chair/unwrap'
    # mesh_texture_file_pairs = {
    #     'backseat.obj':'uv_map_backseat_chair-2_masked.png',
    #     'arms.obj':'uv_map_arms_chair-3_masked.png',
    #     'feet.obj':'uv_map_feet_chair-3_masked.png',
    #     'seat.obj':'uv_map_seat_chair-1_masked.png',
    # }

    # Add all objects and their textures into scene
    for mesh_file,texture_file in mesh_texture_file_pairs.items():
        mesh_path = os.path.join(meshes_dir,mesh_file)
        texture_path = str(p.Path.cwd() / textures_dir / texture_file)
        obj = renderer.load_object(mesh_path)
        uv_path = str(p.Path.cwd() / uv_maps_dir / 'uv_map_{}.png'.format(mesh_file[:-4]))
        # renderer.save_uv_map(obj,save_file=uv_path)
        renderer.apply_texture(obj,texture_path)

    renderer.render()
    # renderer.render_gif(step=90)