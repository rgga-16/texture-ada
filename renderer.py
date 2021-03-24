import os
import pathlib as p
from blender import BlenderRenderer

from PIL import Image


def render_gif(renderer,save_path='./render.gif'):
    still_files = renderer.render_multiple()
    stills = []
    for sf in still_files:
        s = Image.open(sf)
        stills.append(s)
        
    
    stills[0].save(save_path,save_all=True,append_images=stills[1:],duration=200,loop=0)

    for f in still_files:
        os.remove(f)

    return


if __name__ == '__main__':
    renderer = BlenderRenderer()

    # armchair sofa
    meshes_dir = './inputs/shape_samples/armchair sofa'
    textures_dir = 'outputs/output_images/Pyramid2D_with_instnorm/armchair_sofa/[3-11-21 15-00]'
    uv_maps_dir = 'inputs/uv_maps/armchair sofa/unwrap'
    # mesh_texture_file_pairs = {
    #     'backseat.obj':'uv_map_backseat_chair-2_masked.png',
    #     'base.obj':'uv_map_base_chair-1_masked.png',
    #     'left_arm.obj':'uv_map_left_arm_chair-3_masked.png',
    #     'left_foot.obj':'uv_map_left_foot_chair-4_masked.png',
    #     'right_arm.obj':'uv_map_right_arm_chair-3_masked.png',
    #     'right_foot.obj':'uv_map_right_foot_chair-4_masked.png',
    #     'seat.obj':'uv_map_seat_chair-6_masked.png',
    # }

    # mesh_texture_file_pairs = {
    #     'backseat.obj':'uv_map_backseat_chair-2_tiled.png',
    #     'left_arm.obj':'uv_map_left_arm_chair-3_tiled.png',
    #     'right_arm.obj':'uv_map_right_arm_chair-3_tiled.png',
    # }

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

    # lounge sofa
    meshes_dir = './inputs/shape_samples/lounge_sofa'
    textures_dir = 'outputs/output_images/Pyramid2D_with_instnorm/lounge_sofa/[3-17-21 13-22]'
    mesh_texture_file_pairs = {
        'left_arm.obj':'left_arm_uv_chair-3_tiled_256.png',
        'right_arm.obj':'right_arm_uv_chair-3_tiled_256.png',
        'left_backseat.obj':'left_backseat_uv_cobonpue-17_tiled_512.png',
        'mid_backseat.obj':'mid_backseat_uv_chair-2_tiled_256.png',
        'right_backseat.obj':'right_backseat_uv_cobonpue-17_tiled_512.png',
        'left_base.obj':'left_base_uv_cobonpue-80_tiled_512.png',
        'right_base.obj':'right_base_uv_cobonpue-80_tiled_512.png',
        'left_seat.obj':'left_seat_uv_cobonpue-99_tiled_768.png',
        'mid_seat.obj':'mid_seat_uv_chair-2_tiled_256.png',
        'right_seat.obj':'right_seat_uv_cobonpue-99_tiled_768.png',
    }

    # Add all objects and their textures into scene
    for mesh_file,texture_file in mesh_texture_file_pairs.items():
        mesh_path = os.path.join(meshes_dir,mesh_file)
        texture_path = str(p.Path.cwd() / textures_dir / texture_file)
        obj = renderer.load_object(mesh_path)
        # renderer.apply_texture(obj,texture_path)

    renderer.render()
    # render_gif(renderer)