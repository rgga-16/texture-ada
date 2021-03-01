import bpy

import math
import os
import pathlib as p

# import argparse

# from matplotlib import pyplot as plt


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
        bpy.context.scene.render.resolution_x = 512
        bpy.context.scene.render.resolution_y = 512

        self.scene.render.image_settings.file_format = 'PNG'
        bpy.context.scene.render.image_settings.quality = 100
        bpy.context.scene.render.image_settings.color_mode = 'RGBA'

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

    def save_uv_map(self,obj,save_file='//uv_map.png'):
        # Apply Smart uv project from object
        bpy.context.view_layer.objects.active=obj
        obj.select_set(True)
        bpy.ops.object.mode_set(mode="EDIT")

        # bpy.ops.uv.lightmap_pack(PREF_CONTEXT='ALL_FACES')
        # bpy.ops.uv.smart_project()
        bpy.ops.uv.unwrap()
        obj.select_set(False)

        bpy.ops.uv.export_layout(filepath=save_file,mode='PNG',opacity=1.0)
        bpy.ops.object.mode_set(mode="OBJECT")
        bpy.ops.object.select_all(action='DESELECT')
        
        return
        
    
    def render_gif(self,save_path='render.gif',step=15):
        still_files = []
        for obj_angle in range(0,360+step,step):
            for obj in self.objects:
                self.rotate_object(obj,rotation=(
                                    math.radians(0), 
                                    math.radians(obj_angle),
                                    math.radians(0)
                ))
            still_filename = '//render_{}.png'.format(obj_angle)
            self.render(save_path=still_filename)
            still_files.append(still_filename)

        # stills = [load_image(sf) for sf in still_files]
        # save_gif(stills,filename=save_path)

        return

    def render(self,save_path='//render.png'):
        print('Rendering image')
        self.scene.render.filepath = save_path
        bpy.ops.render.render(write_still=True)

    def apply_texture(self,object,texture):
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
        
        if object.data.materials:
            object.data.materials[0]=mat 
        else:
            object.data.materials.append(mat)
        
        # Apply Smart uv project from object
        bpy.context.view_layer.objects.active=object
        object.select_set(True)
        bpy.ops.object.mode_set(mode="EDIT")

        if os.path.basename(texture)=='uv_map_base_chair-1_masked.png':
            bpy.ops.uv.lightmap_pack(PREF_CONTEXT='ALL_FACES')
            obj.select_set(False)
            # pass
        else: 
            bpy.ops.uv.smart_project()
            obj.select_set(False)


    # Render the image

if __name__ == '__main__':
    renderer = BlenderRenderer()

    # meshes_dir = './inputs/shape_samples/armchair sofa'
    meshes_dir = './inputs/shape_samples/office chair'
    textures_dir = 'outputs/output_images/Pyramid2D_with_instnorm/[2-23-21 20-00]'
    uv_maps_dir = 'inputs/uv_maps/office_chair/unwrap'

    # mesh_texture_file_pairs = {
    #     'backseat.obj':'uv_map_backseat_chair-2_masked.png',
    #     # 'base.obj':'uv_map_base_chair-1_masked.png',
    #     # 'left_arm.obj':'uv_map_left_arm_chair-3_masked.png',
    #     # 'left_foot.obj':'uv_map_left_foot_chair-4_masked.png',
    #     # 'right_arm.obj':'uv_map_right_arm_chair-3_masked.png',
    #     # 'right_foot.obj':'uv_map_right_foot_chair-4_masked.png',
    #     # 'seat.obj':'uv_map_seat_chair-6_masked.png',
    # }

    mesh_texture_file_pairs = {
        'backseat.obj':'uv_map_backseat_chair-2_masked.png',
        'arms.obj':'uv_map_base_chair-1_masked.png',
        'feet.obj':'uv_map_left_arm_chair-3_masked.png',
        'seat.obj':'uv_map_left_foot_chair-4_masked.png',
    }

    # Add all objects and their textures into scene
    for mesh_file,texture_file in mesh_texture_file_pairs.items():
        mesh_path = os.path.join(meshes_dir,mesh_file)
        texture_path = str(p.Path.cwd() / textures_dir / texture_file)
        obj = renderer.load_object(mesh_path)
        uv_path = str(p.Path.cwd() / uv_maps_dir / 'uv_map_{}.png'.format(mesh_file[:-4]))
        renderer.save_uv_map(obj,save_file=uv_path)
        # renderer.apply_texture(obj,texture_path)

    # renderer.render()
    # renderer.render_gif(step=90)
