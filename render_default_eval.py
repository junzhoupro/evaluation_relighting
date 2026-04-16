# A simple script that uses blender to render views of a single object by rotation the camera around it.
# Also produces depth map at the same time.

import glob
import os
import random
from math import radians

import bpy
import numpy as np
import sys

DEBUG = False

VIEWS = 12
RESOLUTION = 512
DEPTH_SCALE = 1.4
COLOR_DEPTH = 8
FORMAT = "PNG"
UPPER_VIEWS = True
CIRCLE_FIXED_START = (0, 0, 0)
CIRCLE_FIXED_END = (0.7, 0, 0)


# Get custom args after "--"
argv = sys.argv
argv = argv[argv.index("--") + 1:] if "--" in argv else []

if len(argv) == 0:
    print("No blend file name provided!")
    sys.exit(1)

blend_file = argv[0]
print(f"Running render for: {blend_file}")

# Optionally set output path based on file name
output_name = os.path.splitext(os.path.basename(blend_file))[0]
# bpy.context.scene.render.filepath = f"/your/output/folder/{output_name}.png"

file_name = os.path.splitext(os.path.basename(bpy.data.filepath))[0]
fp = bpy.path.abspath(f"//render_default/{output_name}")

# Get the scene and render settings
scene = bpy.context.scene
render = scene.render

# Render Optimizations
render.use_persistent_data = True

# Background
render.dither_intensity = 0.0
render.film_transparent = True

# Collection name to process
jewelry_collection = bpy.data.collections.get("Jewelry")
body_collection = bpy.data.collections.get("Body")
lighting_collection = bpy.data.collections.get("Lighting")

# Assign a pass index to all objects in the collection
pass_index = 1  # Specify the desired pass index
for obj in jewelry_collection.objects:
    if obj.type == "MESH":  # Only assign to mesh objects
        obj.pass_index = pass_index
# for obj in body_collection.objects:
#     if obj.type == "MESH":  # Only assign to mesh objects
#         obj.pass_index = pass_index

power_settings = {"Light - Key": 250, "Light - Fill": 200, "Light - Back": 50}
for obj in lighting_collection.objects:
    if obj.type == "LIGHT" or obj.type == "MESH":
        obj.hide_viewport = False
        obj.hide_render = False
        obj.visible_camera = False

    if obj.type == "LIGHT" and obj.data.type == "AREA":
        if obj.name in power_settings:
            obj.data.energy = power_settings[obj.name]
            print(f"Set '{obj.name}' to {power_settings[obj.name]}W")

# Get the active view layer dynamically
view_layer = bpy.context.view_layer
view_layer.use_pass_object_index = True  # Enable Object Index render pass

# Set up the environment node in the world shader
world = bpy.context.scene.world
if not world.use_nodes:
    world.use_nodes = True

# Get the environment texture node or create one if it doesn't exist
world_node_tree = world.node_tree
world_nodes = world_node_tree.nodes

# Clear existing nodes (optional for a clean setup)
for node in world_nodes:
    world_nodes.remove(node)

# Add new nodes
output_node = world_nodes.new(type="ShaderNodeOutputWorld")
output_node.location = (200, 0)

bg_node = world_nodes.new(type="ShaderNodeBackground")
bg_node.location = (0, 0)

env_texture_node = world_nodes.new(type="ShaderNodeTexEnvironment")
env_texture_node.location = (-200, 0)

mapping = world_nodes.new(type="ShaderNodeMapping")  # Mapping node for rotation
mapping.location = (-500, 0)

texture_coord = world_nodes.new(type="ShaderNodeTexCoord")  # Texture Coordinate node
texture_coord.location = (-700, 0)

# Connect the nodes
world_node_tree.links.new(bg_node.outputs["Background"], output_node.inputs["Surface"])

# Set up rendering of mask.
scene.use_nodes = True
tree = scene.node_tree
nodes = tree.nodes
links = tree.links

# Clear existing nodes
for node in nodes:
    nodes.remove(node)

# Create Render Layers node
render_layers = nodes.new(type="CompositorNodeRLayers")
render_layers.location = (0, 0)

# Create Output nodes for Mask
mask_output = nodes.new(type="CompositorNodeOutputFile")
mask_output.label = "Mask Output"
mask_output.name = "Mask Output"
mask_output.base_path = f"{fp}/mask"
mask_output.location = (400, -100)

# Add ID Mask node for object or material index masks (optional)
id_mask = nodes.new(type="CompositorNodeIDMask")
id_mask.index = pass_index
id_mask.location = (200, -100)

# Link the nodes
# Link Render Layers Object Index to ID Mask and then to Mask Output
links.new(render_layers.outputs["IndexOB"], id_mask.inputs[0])
set_alpha = nodes.new("CompositorNodeSetAlpha")
links.new(id_mask.outputs[0], set_alpha.inputs[0])
set_alpha.inputs[1].default_value = 1.0  # Fully opaque
links.new(set_alpha.outputs[0], mask_output.inputs[0])


render.resolution_x = RESOLUTION
render.resolution_y = RESOLUTION
render.resolution_percentage = 100
render.image_settings.file_format = str(FORMAT)
render.image_settings.color_depth = str(COLOR_DEPTH)


render.filepath = f"{fp}/{file_name}"
scene.frame_set(1)
tree.nodes["Mask Output"].file_slots[0].path = f"{file_name}_mask_#"
# tree.nodes["Mask Output"].file_slots[0].path = f"{file_name}_mask_1"
bpy.ops.render.render(write_still=True)
