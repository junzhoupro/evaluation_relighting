import bpy
import os

# 设置输入输出路径
# input_file = "hdrs_chosen/cinema_hall_4k.exr"
# output_file = "hdrs_png/cinema_hall_4k.png"

# input_file = "hdrs_chosen/indoor_pool_4k.exr"
# output_file = "hdrs_png/indoor_pool_4k.png"

input_file = "hdrs_chosen/industrial_pipe_and_valve_01_4k.exr"
output_file = "hdrs_png/industrial_pipe_and_valve_01_4k.png"

# 1. 加载图片
img = bpy.data.images.load(os.path.abspath(input_file))
width, height = img.size

# 2. 强制设置渲染分辨率匹配图片尺寸（解决“不是全景”的问题）
bpy.context.scene.render.resolution_x = width
bpy.context.scene.render.resolution_y = height
bpy.context.scene.render.resolution_percentage = 100

# 3. 设置合成节点
bpy.context.scene.use_nodes = True
tree = bpy.context.scene.node_tree
tree.nodes.clear()

node_in = tree.nodes.new(type='CompositorNodeImage')
node_in.image = img
node_out = tree.nodes.new(type='CompositorNodeComposite')
tree.links.new(node_in.outputs[0], node_out.inputs[0])

# 4. 设置色彩管理 (这里决定了颜色是否和 Blender 渲染结果一致)
bpy.context.scene.display_settings.display_device = 'sRGB'
bpy.context.scene.view_settings.view_transform = 'AgX' # 如果是旧版建议用 'Filmic'
bpy.context.scene.view_settings.look = 'None'

# 5. 输出设置
bpy.context.scene.render.image_settings.file_format = 'PNG'
bpy.context.scene.render.image_settings.color_mode = 'RGB' # Paper 通常不需要 Alpha
bpy.context.scene.render.filepath = os.path.abspath(output_file)

# 6. 执行渲染转换
bpy.ops.render.render(write_still=True)