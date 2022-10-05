import os
import pathlib as p
import blender
import renderer as r
from blender import BlenderRenderer
import math 

if __name__ == '__main__':
    images = [
    'cobonpue-8-1.png', #red flower
    '35.ch-vito-ale-hi-natwh-1.png', #rectangular basket swing thing
    'cobonpue-92-1.png', #twirled chair
    'selma-19.png', #semicircle folding chair,
    'selma-88.png', #circular chair
    ]

    model_dirs = [ 
        'armchair_5',
        'dining_chair_1',
        'lounge_sofa_2',
        'office_chair_3'
    ]
    renderer = BlenderRenderer()
    renderer.move_camera((0,1.0,3.80),None)

    dir = './outputs/output_models'
    for d in os.listdir(dir):
        
        shapes_dir = os.path.join(dir,d)
        if os.path.isfile(shapes_dir):
            continue

        for im in images:
            curr_dir = os.path.join(shapes_dir,im)
            render_dir = os.path.join('./outputs/renders/blender/output_models',d,im)
            try:
                os.makedirs(render_dir,exist_ok=True)
            except FileExistsError:
                pass
            for m in model_dirs:
                obj_path = os.path.join(curr_dir,f'{m}.obj')
                pcd_path = os.path.join(curr_dir,f'{m}.ply')
                real_pcd_path = os.path.join(curr_dir,f'{m}_real.ply')
                
                pcd = renderer.load_ply(pcd_path)
                renderer.paint_vertex_with_spheres(pcd)
                renderer.rotate_object(pcd,(math.radians(180), math.radians(0),math.radians(0)))
                save_path = os.path.join(render_dir,f'{m}_pcd.png')
                
                # r.render_image(renderer,rotation=(math.radians(180), math.radians(160),math.radians(0)),save_path=save_path)
                r.render_gif(renderer,os.path.join(render_dir,f'{m}_pcd.gif'))
                renderer.clear()

                pcd = renderer.load_ply(real_pcd_path)
                renderer.paint_vertex_with_spheres(pcd)
                save_path = os.path.join(render_dir,f'{m}_pcd_real.png')
                # r.render_image(renderer,rotation=(math.radians(0), math.radians(160),math.radians(0)),save_path=save_path)
                r.render_gif(renderer,os.path.join(render_dir,f'{m}_pcd_real.gif'))
                renderer.clear()

                mesh=renderer.load_object(obj_path)
                renderer.recalculate_normals(mesh)
                renderer.rotate_object(mesh,(math.radians(180), math.radians(0),math.radians(0)))
                save_path = os.path.join(render_dir,f'{m}_mesh.png')
                # r.render_image(renderer,rotation=(math.radians(180), math.radians(160),math.radians(0)),save_path=save_path)
                r.render_gif(renderer,os.path.join(render_dir,f'{m}_mesh.gif'))
                renderer.clear()
                print()


    # # INSERT RENDERING MODULE HERE
    # #######################################
    # for chair_dir in os.listdir(chair_dirs): 
    #     if os.path.isfile(os.path.join(chair_dirs,chair_dir)):
    #         continue
    #     mesh_dir = os.path.join(chair_dirs,chair_dir)
    #     for m in os.listdir(mesh_dir):
    #         mesh_path = os.path.join(mesh_dir,m)
    #         if os.path.isdir(mesh_path):
    #             continue
    #         texture_val = ctp['base']
    #         for c in ctp.keys():
    #             if c in m:
    #                 texture_val = ctp[c]
    #                 break 
    #         for t in os.listdir(texture_dir):
    #             if texture_val.casefold() == t.casefold():
    #                 text_path = str(p.Path.cwd() / texture_dir / t)
    #                 # text_path = os.path.join(texture_dir,t)
    #                 break 
    #         assert text_path is not None 
    #         obj = renderer.load_object(mesh_path)
    #         renderer.recalculate_normals(obj)
    #         renderer.bake_texture(obj,unwrap_method,text_path)
        
    #     save_dir = f'./outputs/renders/blender/{os.path.splitext(os.path.basename(chair_textures))[0]}'
    #     try:
    #         os.makedirs(save_dir,exist_ok=True)
    #     except FileExistsError:
    #         pass
    #     save_path = os.path.join(save_dir,f'{os.path.basename(chair_dir)}.png')
    #     # r.render_gif(renderer,save_path=save_path)
    #     r.render_image(renderer,rotation_angle=-120,save_path=save_path)
    #     renderer.clear()
  