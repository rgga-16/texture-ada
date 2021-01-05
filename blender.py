import bpy
import math



class Renderer():
    def __init__(self):
        # Add scene
        self.create_scene()

        # Add a camera
        self.setup_camera()
        # Add lights    
        self.setup_lights()

        # Load objects
        self.load_object()
        # Load respective texture images
        # Add texture to object material
        # Render

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

    def setup_camera(self):
        bpy.ops.object.camera_add()
        self.camera = bpy.data.objects['Camera']
        self.camera.location=(0.0,0.0,2.0)
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
    
    def load_object(self,texture_path='//textmap_1024.png',mesh_path='./inputs/shape_samples/armchair sofa/backseat.obj',):
        bpy.ops.import_scene.obj(filepath=mesh_path)
        obj = bpy.context.selected_objects[0]

        obj.rotation_euler[0]=math.radians(0)
        obj.rotation_euler[1]=math.radians(90)

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
        
        self.save_uv_map(obj)


        
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
    
    renderer.render()

