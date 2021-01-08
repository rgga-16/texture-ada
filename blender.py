import bpy
import math
import os

# import utils  
# from defaults import DEFAULTS as D 

class Renderer():
    def __init__(self):
        # Add scene
        self.create_scene()

        # Add a camera
        self.setup_camera()
        # Add lights    
        self.setup_lights()


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

    def setup_camera(self,location=(0.0,0.3,1.3),
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
        lamp_object.location = (15.0, 0.0, 15.0)
        # And finally select it make active
        lamp_object.select_set(state=True)
        # self.scene.objects.active = lamp_object
        bpy.context.view_layer.objects.active = lamp_object
    
    def load_object(self,mesh_path,texture_path):
        bpy.ops.import_scene.obj(filepath=mesh_path)
        obj = bpy.context.selected_objects[0]

        obj.rotation_euler[0]=math.radians(0)
        obj.rotation_euler[1]=math.radians(0)

        image = bpy.data.images.load(texture_path)

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
        
        # self.save_uv_map(obj)


        
        return

    def save_uv_map(self,obj,save_file='//uv_map.png'):
        bpy.context.view_layer.objects.active=obj
        bpy.ops.object.mode_set(mode="EDIT")
        bpy.ops.mesh.select_all(action='SELECT')
        bpy.ops.uv.smart_project()

        bpy.ops.uv.export_layout(filepath=save_file,mode='PNG',opacity=1.0)
        
        bpy.ops.object.select_all(action='DESELECT')
        bpy.ops.object.mode_set(mode="OBJECT")



    def render(self):
        print('Rendering')
        self.scene.render.filepath = '//render.png'
        bpy.ops.render.render(write_still=True)




if __name__ == '__main__':
    renderer = Renderer()

    # meshes_dir = os.path.join(D.MESHES_DIR.get(), D.MESH_DIR.get())
    meshes_dir = './inputs/shape_samples/armchair sofa'
    textures_dir = './inputs/texture_maps/texture_network'

    mesh_texture_file_pairs = {
        'backseat.obj':'output_chair-2_cropped.png',
        'base.obj':'output_chair-4_cropped.png',
        'left_arm.obj':'output_chair-5_cropped.png',
        'left_foot.obj':'output_chair-3_cropped.png',
        'right_arm.obj':'output_chair-5_cropped.png',
        'right_foot.obj':'output_chair-3_cropped.png',
        'seat.obj':'output_chair-1_cropped.png',
    }


    for mesh_file,texture_file in mesh_texture_file_pairs.items():
        mesh_path = os.path.join(meshes_dir,mesh_file)
        # texture_path = textures_dir + '/' + texture_file
        texture_path = '//' + texture_file
        renderer.load_object(mesh_path,texture_path)
        
    renderer.render()

