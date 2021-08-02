import os
import pathlib as p
import blender
import renderer as r
from blender import BlenderRenderer
import json 

from PIL import Image

if __name__ == '__main__':
    chair_textures_ = [
                        # './inputs/uv_style_pairs/chair_texture_1.json',
                        # './inputs/uv_style_pairs/chair_texture_2.json',
                        # './inputs/uv_style_pairs/chair_texture_3.json',
                        './inputs/uv_style_pairs/chair_single_texture.json',
                        # './inputs/uv_style_pairs/chair_multi_texture.json',
                        ]
    # selected = ['2.png','12.png','23.png']
    selected = ['23.png']
    
    
    renderer = BlenderRenderer()
    for chair_textures in chair_textures_:
        data = json.load(open(chair_textures))
        chair_dirs = data['chair_dir']
        texture_dirs = data['texture_dir']
        unwrap_method='cube_project'

        for select in selected:
            ctp = {
                "backseat":f"{select}",
                "arm":f"{select}",
                "foot":f"{select}",
                "feet":f"{select}",
                "base":f"{select}",
                "body":f"{select}",
                "seat":f"{select}",
                "armrest":f"{select}"
            }
    
            for chair_dir in os.listdir(chair_dirs): 
                if os.path.isfile(os.path.join(chair_dirs,chair_dir)):
                    continue
                mesh_dir = os.path.join(chair_dirs,chair_dir)

                for texture_dir in texture_dirs:
                    for m in os.listdir(mesh_dir):
                        mesh_path = os.path.join(mesh_dir,m)
                        if os.path.isdir(mesh_path):
                            continue
                        texture_val = ctp['base']
                        # for c in ctp.keys():
                        #     if c in m:
                        #         texture_val = ctp[c]
                        #         break 
                        # for t in os.listdir(texture_dir):
                        #     if os.path.isdir(os.path.join(texture_dir,t)):
                        #         continue
                        #     if (os.path.splitext(os.path.join(texture_dir,t))[1] not in ['.png']):
                        #         continue 
                        #     if texture_val.casefold() == t.casefold():
                        #         text_path = str(p.Path.cwd() / texture_dir / t)
                        #         break 
                        text_path = str(p.Path.cwd() / texture_dir / texture_val)
                        assert text_path is not None 
                        obj = renderer.load_object(mesh_path)
                        renderer.recalculate_normals(obj)
                        renderer.bake_texture(obj,unwrap_method,text_path)
                
                    save_dir = f'./outputs/renders/blender/{os.path.splitext(os.path.basename(chair_textures))[0]}/{select}'
                    try:
                        os.makedirs(save_dir,exist_ok=True)
                    except FileExistsError:
                        pass
                    save_path_gif = os.path.join(save_dir,f'{os.path.basename(texture_dir)}.gif')
                    save_path_png = os.path.join(save_dir,f'{os.path.basename(texture_dir)}.png')
                    r.render_image(renderer,rotation_angle=-120,save_path=save_path_png)
                    r.render_gif(renderer,save_path=save_path_gif) 
                    renderer.clear()
        