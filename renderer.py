import os
import pathlib as p
from blender import BlenderRenderer

from PIL import Image
import math 

def render_image(renderer,rotation=None,save_path='./render.png'):

    
    # if rotation_angle:
    #     rotation=(math.radians(0), math.radians(rotation_angle),math.radians(0))

    path = renderer.render_single(rotation)
    render_file = Image.open(path)
    
    dir_name = os.path.dirname(save_path)
    try:
        os.makedirs(dir_name,exist_ok=True)
    except FileExistsError:
        pass
    render_file.save(save_path)

    os.remove(path)

    return 

def gen_frame(im):
    alpha = im.getchannel('A')

    # Convert the image into P mode but only use 255 colors in the palette out of 256
    im = im.convert('RGB').convert('P', palette=Image.ADAPTIVE, colors=255)

    # Set all pixel values below 128 to 255 , and the rest to 0
    mask = Image.eval(alpha, lambda a: 255 if a <=128 else 0)

    # Paste the color of index 255 and use alpha as a mask
    im.paste(255, mask)

    # The transparency index is 255
    im.info['transparency'] = 255

    return im

def render_gif(renderer,save_path='./render.gif'):
    still_files = renderer.render_multiple()
    stills = []
    for sf in still_files:
        s = Image.open(sf)
        s = gen_frame(s)
        stills.append(s)
    
    dir_name = os.path.dirname(save_path)
    try:
        os.makedirs(dir_name,exist_ok=True)
    except FileExistsError:
        pass
        
    
    stills[0].save(save_path,save_all=True,append_images=stills[1:],duration=200,loop=0,disposal=2)

    for f in still_files:
        os.remove(f)

    return


def render_model(meshes_dir,textures_dir, mesh_texture_file_pairs,uv_test_sizes):

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
    meshes_dir = './inputs/3d_models/shapenet/dining_chair_3'
    textures_dir = 'outputs/output_images/[05-06-21 17-19-51]/ProposedModel'
    uv_maps_dir = 'inputs/uv_maps/dining_chair_3_1/cube_project'
    mesh_texture_file_pairs = {
        "backseat.obj":"backseat_uv.png",
        "left_arm.obj":"left_arm_uv.png",
        "right_arm.obj":"right_arm_uv.png",
        "seat.obj":"seat_uv.png",
        "body.obj":"body_uv.png"
    }

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
        render_gif(renderer,save_path='./outputs/renders/blender/office_chair/[04-21-21 06-17-40]/render_officechair_{}.gif'.format(size))