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
    
    dir_name = os.path.dirname(save_path)
    try:
        os.mkdir(dir_name)
    except FileExistsError:
        pass
        
    
    stills[0].save(save_path,save_all=True,append_images=stills[1:],duration=200,loop=0)

    for f in still_files:
        os.remove(f)

    return


if __name__ == '__main__':
    

    # armchair sofa
    # meshes_dir = './inputs/shape_samples/armchair sofa'
    # textures_dir = 'outputs/output_images/Pyramid2D_with_instnorm/armchair_sofa/[3-11-21 15-00]'
    # uv_maps_dir = 'inputs/uv_maps/armchair sofa/unwrap'
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
    meshes_dir = './inputs/shape_samples/office_chair'
    textures_dir = 'outputs/output_images/[04-21-21 12-07-53]'
    uv_maps_dir = 'inputs/uv_maps/office chair/cube_project'
    mesh_texture_file_pairs = {
        "backseat.obj":"backseat_uv.png",
        "arms.obj":"arms_uv.png",
        "feet.obj":"feet_uv.png",
        "seat.obj":"seat_uv.png"
    }
    
    # lounge sofa
    # meshes_dir = './inputs/shape_samples/lounge_sofa'
    # # textures_dir = 'outputs/output_images/Pyramid2D_with_instnorm/lounge_sofa/[04-08-21 09-00-56] Johnson (w-o fg weight)'
    # textures_dir = 'outputs/output_images/[04-14-21 06-28-05] Ulyanov-Adain - Multistyle (w-o IN layers) 2'
    # mesh_texture_file_pairs = {
    #     'left_arm.obj':'left_arm_uv.png',
    #     'right_arm.obj':'right_arm_uv.png',
    #     'left_backseat.obj':'left_backseat_uv.png',
    #     'mid_backseat.obj':'mid_backseat_uv.png',
    #     'right_backseat.obj':'right_backseat_uv.png',
    #     'left_base.obj':'left_base_uv.png',
    #     'right_base.obj':'right_base_uv.png',
    #     'left_seat.obj':'left_seat_uv.png',
    #     'mid_seat.obj':'mid_seat_uv.png',
    #     'right_seat.obj':'right_seat_uv.png',
    # }

    # # round table
    # meshes_dir = './inputs/shape_samples/round_table/models'
    # textures_dir = 'outputs/output_images/Pyramid2D_with_instnorm/round_table/[03-24-21 05-32-44] round table - uv map config 1A'
    # mesh_texture_file_pairs = {
    #     'tabletop.obj':'tabletop_uv.png',
    #     'botleft_leg.obj':'botleft_leg_uv.png',
    #     'botright_leg.obj':'botright_leg_uv.png',
    #     'topleft_leg.obj':'topleft_leg_uv.png',
    #     'topright_leg.obj':'topright_leg_uv.png',
    # }

    # for size in [256, 512, 768, 1024]:
    for size in [256]:
        renderer = BlenderRenderer()
        # Add all objects and their textures into scene
        for mesh_file,texture_file in mesh_texture_file_pairs.items():
            mesh_path = os.path.join(meshes_dir,mesh_file)
            texture_path = str(p.Path.cwd() / textures_dir / '{}_{}.png'.format(texture_file,size))
            obj = renderer.load_object(mesh_path)
            renderer.recalculate_normals(obj)
            renderer.apply_texture(obj,'CUBE_PROJECT',texture_path)

        # renderer.render(save_path='//render_round_table_colorless_{}.png'.format(size))
        render_gif(renderer,save_path='./outputs/renders/blender/office_chair/[04-21-21 12-07-53]/render_officechair_{}.gif'.format(size))