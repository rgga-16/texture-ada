import bpy

import math
import os
import pathlib as p

# from PIL import Image

# import utils  
# from defaults import DEFAULTS as D 

# Loads image
def load_image(filename):
    img = Image.open(filename).convert("RGB")
    return img

def save_gif(images:list,filename='output.gif'):
    images[0].save(filename,save_all=True,append_images=images[1:],loop=0,optimize=False,duration=75)

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
        bpy.context.scene.render.resolution_x = 1024
        bpy.context.scene.render.resolution_y = 1024

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
        obj = bpy.context.selected_objects[0]

        obj.rotation_euler = (math.radians(0),
                            math.radians(90),
                            math.radians(0))

        image = bpy.data.images.load(texture_path,check_existing=True)

        mat = bpy.data.materials.new(name='Material')
        mat.use_nodes=True     
        bsdf = mat.node_tree.nodes["Principled BSDF"]
        texture_image = mat.node_tree.nodes.new('ShaderNodeTexImage')
        texture_image.image = image 
        mat.node_tree.links.new(bsdf.inputs['Base Color'],texture_image.outputs['Color'])
        
        if obj.data.materials:
            obj.data.materials[0]=mat 
        else:
            obj.data.materials.append(mat)
        
        self.objects.append(obj)
        
        # self.save_uv_map(obj)

        return obj
    
    def rotate_object(self,obj,rotation):
        obj.rotation_euler = rotation
        return obj

    def save_uv_map(self,obj,save_file='//uv_map.png'):
        bpy.context.view_layer.objects.active=obj
        bpy.ops.object.mode_set(mode="EDIT")
        bpy.ops.mesh.select_all(action='SELECT')
        bpy.ops.uv.smart_project()

        bpy.ops.uv.export_layout(filepath=save_file,mode='PNG',opacity=1.0)
        
        bpy.ops.object.select_all(action='DESELECT')
        bpy.ops.object.mode_set(mode="OBJECT")
    
    def render_gif(self,save_path='render.gif'):
        step=15
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
        image = bpy.data.images.load(texture_path,check_existing=True)

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
        bpy.ops.object.mode_set(mode="EDIT")
        bpy.ops.mesh.select_all(action='SELECT')
        bpy.ops.uv.smart_project()

    # Render the image

if __name__ == '__main__':
    renderer = BlenderRenderer()

    meshes_dir = './inputs/shape_samples/armchair sofa'
    textures_dir = 'inputs/texture_maps/texture_network/ulyanov_texturenet/instnorm'

    mesh_texture_file_pairs = {
        'backseat.obj':'uv_chair2.png',
        # 'backseat.obj':'output_chair-2_cropped.png',
        # 'base.obj':'output_chair-4_cropped.png',
        # 'left_arm.obj':'output_chair-5_cropped.png',
        # 'left_foot.obj':'output_chair-3_cropped.png',
        # 'right_arm.obj':'output_chair-5_cropped.png',
        # 'right_foot.obj':'output_chair-3_cropped.png',
        # 'seat.obj':'output_chair-1_cropped.png',
    }

    # Add all objects and their textures into scene
    for mesh_file,texture_file in mesh_texture_file_pairs.items():
        mesh_path = os.path.join(meshes_dir,mesh_file)
        texture_path = str(p.Path.cwd() / textures_dir / texture_file)
        obj = renderer.load_object(mesh_path)
        renderer.apply_texture(obj,texture_path)
    renderer.render()
        
    # renderer.render_gif()
