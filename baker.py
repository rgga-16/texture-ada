import os
import pathlib as p
import blender
import renderer as r
from blender import BlenderRenderer
import json ,math

from PIL import Image

if __name__ == '__main__':
    chair_textures_ = [
                        # './inputs/uv_style_pairs/chair_texture_1.json',
                        # './inputs/uv_style_pairs/chair_texture_2.json',
                        # './inputs/uv_style_pairs/chair_texture_3.json',
                        # './inputs/uv_style_pairs/chair_texture_4.json',
                        './inputs/uv_style_pairs/table_texture_1.json',
                        './inputs/uv_style_pairs/table_texture_2.json',
                        './inputs/uv_style_pairs/table_texture_single.json',
                        './inputs/uv_style_pairs/table_texture_4.json',
                        ]
    
    renderer = BlenderRenderer()
    for chair_textures in chair_textures_:
        print(f'Applying texture config: {os.path.basename(chair_textures)}')
        data = json.load(open(chair_textures))
        chair_dirs = data['chair_dir']
        texture_dir = data['texture_dir']
        ctp = data['chair_texture_pairs']
        unwrap_method='cube_project'

    # INSERT RENDERING MODULE HERE
    #######################################
    
        for chair_dir in os.listdir(chair_dirs): 
            if os.path.isfile(os.path.join(chair_dirs,chair_dir)):
                continue
            mesh_dir = os.path.join(chair_dirs,chair_dir)
            for m in os.listdir(mesh_dir):
                mesh_path = os.path.join(mesh_dir,m)
                if os.path.isdir(mesh_path):
                    continue
                texture_val = ctp['base']
                for c in ctp.keys():
                    if c in m:
                        texture_val = ctp[c]
                        break 
                for t in os.listdir(texture_dir):
                    if texture_val.casefold() == t.casefold():
                        text_path = str(p.Path.cwd() / texture_dir / t)
                        # text_path = os.path.join(texture_dir,t)
                        break 
                assert text_path is not None 
                obj = renderer.load_object(mesh_path)
                renderer.recalculate_normals(obj)
                renderer.bake_texture(obj,unwrap_method,text_path)
            
            save_dir = f'./outputs/renders/blender/{os.path.splitext(os.path.basename(chair_textures))[0]}'
            try:
                os.makedirs(save_dir,exist_ok=True)
            except FileExistsError:
                pass
            save_path_gif = os.path.join(save_dir,f'{os.path.basename(chair_dir)}_topview.gif')
            save_path_png = os.path.join(save_dir,f'{os.path.basename(chair_dir)}_topview.png')
            renderer.setup_camera((0.0,0.7,1.3),(math.radians(-30),math.radians(0),math.radians(0)))
            renderer. move_lamp((0.0,0.7,0.0))
            r.render_image(renderer,rotation_angle=-120,save_path=save_path_png)
            r.render_gif(renderer,save_path=save_path_gif) 
            renderer.clear()
    