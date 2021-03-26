import bpy

import math
import os
import pathlib as p

import argparse
import logging 

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


def unwrap_method(method:str):
        if method.lower()=='SMART_PROJECT'.lower():
            bpy.ops.uv.smart_project()
        elif method.lower()=='UNWRAP'.lower():
            bpy.ops.uv.unwrap()
        elif method.lower()=='CUBE_PROJECT'.lower():
            bpy.ops.uv.cube_project(cube_size=1.0, 
                        correct_aspect=True, 
                        clip_to_bounds=True, 
                        scale_to_bounds=True) 
        elif method.lower()=='CYLINDER_PROJECT'.lower():
            bpy.ops.uv.cylinder_project(direction='VIEW_ON_EQUATOR', 
                            align='POLAR_ZX', radius=1.0, correct_aspect=True, 
                            clip_to_bounds=True, scale_to_bounds=True)
        elif method.lower()=='LIGHTMAP_PACK'.lower():
            bpy.ops.uv.lightmap_pack(PREF_CONTEXT='ALL_FACES')
        else: 
            logging.exception('ERROR: Invalid unwrapping method.')
        


class BlenderRenderer():

    def __init__(self):
        # Add scene
        self.create_scene()
        # Add a camera
        self.setup_camera()
        # Add lights    
        self.setup_lights()

        self.objects=[]


    def create_scene(self):
        self.scene = bpy.context.scene
        self.scene.render.resolution_x = 512
        self.scene.render.resolution_y = 512

        self.scene.render.image_settings.file_format = 'PNG'
        self.scene.render.image_settings.quality = 100
        self.scene.render.image_settings.color_mode = 'RGBA'

        self.scene.render.resolution_percentage = 100
        self.scene.render.use_border = False
        # self.scene.render.alpha_mode = 'TRANSPARENT'

        bpy.ops.object.select_all(action='SELECT')
        bpy.ops.object.delete(use_global=False)

    def setup_camera(self,
                    location=(0.0,0.3,1.3),
                    rotation=(math.radians(-15),math.radians(0),math.radians(0))):

        bpy.ops.object.camera_add()
        self.camera = bpy.data.objects['Camera']
        self.camera.location=location
        self.camera.rotation_euler = rotation
        # Add camera to scene
        bpy.context.scene.camera = self.camera
    
    def setup_lights(self):
        lamp_data = bpy.data.lights.new(name="New Lamp", type='SUN')
        lamp_data.energy = 1
        # Create new object with our lamp datablock
        lamp_object = bpy.data.objects.new(name="New Lamp", object_data=lamp_data)
        # Link lamp object to the scene so it'll appear in this scene
        self.scene.collection.objects.link(lamp_object)
        # Place lamp to a specified location
        lamp_object.location = (0.0,0.0,0.0)
        # And finally select it make active
        lamp_object.select_set(state=True)
        # self.scene.objects.active = lamp_object
        bpy.context.view_layer.objects.active = lamp_object
        return
    
    def load_object(self,mesh_path):
        bpy.ops.import_scene.obj(filepath=mesh_path)
        mesh_file = os.path.basename(mesh_path)
        selected = bpy.context.selected_objects

        obj = bpy.context.selected_objects.pop()

        obj.rotation_euler = (math.radians(0),
                            math.radians(270),
                            math.radians(0))

        self.objects.append(obj)


        return obj
    
    def rotate_object(self,obj,rotation):
        obj.rotation_euler = rotation
        return obj
    
    

    def save_uv_map(self,obj,unwrap_method_,save_file='//uv_map.png'):
        # Apply Smart uv project from object
        bpy.context.view_layer.objects.active=obj
        obj.select_set(True)
        selected = bpy.context.selected_objects
        bpy.ops.object.mode_set(mode="EDIT")

        unwrap_method(unwrap_method_)

        obj.select_set(False)

        bpy.ops.uv.export_layout(filepath=save_file,mode='PNG',size=(256,256),opacity=1.0)
        bpy.ops.object.mode_set(mode="OBJECT")
        bpy.ops.object.select_all(action='DESELECT')
        
        return
        
    
    def render_multiple(self,step=15):
        still_files = []
        for obj_angle in range(0,360+step,step):
            for obj in self.objects:
                self.rotate_object(obj,rotation=(math.radians(0), math.radians(obj_angle),math.radians(0)))

            still_filename = 'render_{}.png'.format(obj_angle)
            save_path = str(p.Path.cwd() / still_filename)
            self.render(save_path=save_path)
            still_files.append(still_filename)

        return still_files

    def render(self,save_path='//render.png'):
        print('Rendering image')
        bpy.context.scene.render.film_transparent = True
        self.scene.render.filepath = save_path
        bpy.ops.render.render(write_still=True)

    def apply_texture(self,object,unwrap_method_,texture):
        # Load texture as texture_image node
        image = bpy.data.images.load(texture,check_existing=True)

        # Load object material's Principal BSDF node
        mat = bpy.data.materials.new(name='Material')
        mat.use_nodes=True     
        bsdf = mat.node_tree.nodes["Principled BSDF"]

        # Link texture_image's node Color to BSDF node BaseColor
        texture_image = mat.node_tree.nodes.new('ShaderNodeTexImage')
        texture_image.image = image 
        mat.node_tree.links.new(bsdf.inputs['Base Color'],texture_image.outputs['Color'])
        object.data.materials.clear()

        object.data.materials.append(mat)
        
        bpy.context.view_layer.objects.active=object
        object.select_set(True)
        bpy.ops.object.mode_set(mode="EDIT")

        unwrap_method(unwrap_method_)
        object.select_set(False)

        # if os.path.basename(texture)=='uv_map_base_chair-1_masked.png':
        #     bpy.ops.uv.unwrap()
        #     object.select_set(False)
        # else: 
        #     bpy.ops.uv.smart_project()
        #     # bpy.ops.uv.unwrap()
        #     object.select_set(False)

        bpy.ops.object.mode_set(mode="OBJECT")
        bpy.ops.object.select_all(action='DESELECT')

if __name__ == '__main__':
    renderer = BlenderRenderer()

    mesh = "model_modified.obj"
    uv = "hello"
    mode = "MULTI"
    unwrap_method_='smart_project'
    mesh_dir = "inputs/shape_samples/office_chair"
    uv_dir = "inputs/uv_maps/office_chair/{}".format(unwrap_method_)

    if mode=='SINGLE':
        mesh_path =os.path.join(mesh_dir,mesh)
        
        renderer.load_object(mesh_path)
        selected = [obj_ for obj_ in bpy.context.selected_objects if obj_.type=='MESH']

        bpy.ops.object.mode_set(mode="OBJECT")
        bpy.ops.object.select_all(action='DESELECT')

        try:
            os.mkdir(uv_dir)
        except FileExistsError:
            pass

        for obj in selected:
            name = obj.name
            uv_path = os.path.join(uv_dir,'{}_uv.png'.format(name))
            
            renderer.save_uv_map(obj,unwrap_method_,save_file=uv_path)
    else:
        for mesh_file in os.listdir(mesh_dir):
            ext = os.path.splitext(mesh_file)[-1].lower()
            if ext in ['.obj']:
                mesh_path = os.path.join(mesh_dir,mesh_file)
                obj = renderer.load_object(mesh_path)
                uv_path = str(p.Path.cwd() / uv_dir / '{}_uv.png'.format(mesh_file[:-4]))
                renderer.save_uv_map(obj,unwrap_method_,save_file=uv_path)