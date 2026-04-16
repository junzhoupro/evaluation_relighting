import os
import sys
import bpy

# ===========================
# 读取命令行参数
# ===========================
argv = sys.argv
argv = argv[argv.index("--") + 1:] if "--" in argv else []

if len(argv) == 0:
    print("No blend file name provided!")
    sys.exit(1)

blend_file = argv[0]
print(f"Running render for: {blend_file}")

# 解析 --hdr-name 参数
hdr_name = None
for i in range(len(argv)):
    if argv[i] == "--hdr-name" and i + 1 < len(argv):
        hdr_name = argv[i + 1]
        break

if hdr_name is None:
    print("No --hdr-name provided!")
    sys.exit(1)

print(f"Received HDR name: {hdr_name}")

# 设置 HDRI 路径（根据 hdr_name 拼接）
hdr1_path = f"/home/jz927/Documents/riggedhands/rigged-hand/hdrs2/{hdr_name}.exr"

# 输出路径
output_name = os.path.splitext(os.path.basename(blend_file))[0]
file_name = os.path.splitext(os.path.basename(bpy.data.filepath))[0]
fp = bpy.path.abspath(f"//{hdr_name}/{output_name}")

# ===========================
# 渲染参数设置
# ===========================
RESOLUTION = 512
COLOR_DEPTH = 8
FORMAT = "PNG"

scene = bpy.context.scene
render = scene.render

render.resolution_x = RESOLUTION
render.resolution_y = RESOLUTION
render.resolution_percentage = 100
render.image_settings.file_format = str(FORMAT)
render.image_settings.color_depth = str(COLOR_DEPTH)
render.use_persistent_data = True
render.dither_intensity = 0.0
render.film_transparent = False

scene.use_nodes = True
bpy.context.view_layer.use_pass_object_index = True
bpy.context.scene.frame_set(1)


# ===========================
# 禁用所有场景灯光
# ===========================
def disable_all_lights():
    # 禁用所有灯光对象
    for obj in bpy.context.scene.objects:
        if obj.type == 'LIGHT':
            obj.hide_render = True  # 禁用渲染
            print(f"禁用灯光: {obj.name}")
    
    # 确保使用纯HDRI照明
    world = bpy.context.scene.world
    if world:
        world.use_nodes = True

def disable_lighting_collection():
    lighting_collection = bpy.data.collections.get("Lighting")
    if lighting_collection:
        lighting_collection.hide_render = True  # 整个集合在渲染时隐藏
        print("禁用整个 Lighting 集合")

# ===========================
# Helper: 设置 HDRI 环境
# ===========================
def set_hdri(hdri_path):
    world = bpy.context.scene.world
    if not world.use_nodes:
        world.use_nodes = True
    nodes = world.node_tree.nodes
    links = world.node_tree.links

    # 清空原有节点
    for node in nodes:
        nodes.remove(node)

    # 创建节点
    node_tex = nodes.new("ShaderNodeTexEnvironment")
    node_tex.image = bpy.data.images.load(hdri_path)
    node_bg = nodes.new("ShaderNodeBackground")
    node_out = nodes.new("ShaderNodeOutputWorld")

    # 连接节点
    links.new(node_tex.outputs["Color"], node_bg.inputs["Color"])
    links.new(node_bg.outputs["Background"], node_out.inputs["Surface"])

# ===========================
# 设置 Jewelry mask pass index
# ===========================
def assign_mask_index(collection, index):
    for obj in collection.objects:
        if obj.type == "MESH":
            obj.pass_index = index

# ===========================
# 渲染 HDR1 + mask
# ===========================
set_hdri(hdr1_path)
disable_all_lights()  # 添加这行
disable_lighting_collection()
jewelry_collection = bpy.data.collections.get("Jewelry")
assign_mask_index(jewelry_collection, 1)

tree = scene.node_tree
nodes = tree.nodes
links = tree.links
nodes.clear()

# Render Layers
render_layers = nodes.new("CompositorNodeRLayers")

# 保存 RGB 输出
# color_output = nodes.new("CompositorNodeOutputFile")
# color_output.base_path = f"{fp}/rgb"
# color_output.file_slots[0].path = f"{file_name}_rgb_#"
# links.new(render_layers.outputs["Image"], color_output.inputs[0])

# 保存 mask 输出
id_mask = nodes.new("CompositorNodeIDMask")
id_mask.index = 1
mask_output = nodes.new("CompositorNodeOutputFile")
mask_output.base_path = f"{fp}/mask"
mask_output.file_slots[0].path = f"{file_name}_mask_#"
# mask_output.file_slots[0].path = f"{file_name}_mask_???"

set_alpha = nodes.new("CompositorNodeSetAlpha")
links.new(render_layers.outputs["IndexOB"], id_mask.inputs[0])
links.new(id_mask.outputs[0], set_alpha.inputs[0])
set_alpha.inputs[1].default_value = 1.0
links.new(set_alpha.outputs[0], mask_output.inputs[0])

# 渲染
render.filepath = f"{fp}/{file_name}"
bpy.ops.render.render(write_still=True)
