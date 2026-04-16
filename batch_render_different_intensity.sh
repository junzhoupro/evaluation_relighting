#!/bin/bash

BLENDER_PATH=/snap/bin/blender
PY_SCRIPT=render_default_different_intensity.py

# 所有 blend 文件所在目录（可以一次处理多个目录）
# BLEND_DIRS=(
#     ./aged-hand_choose_rotate
#     ./aged-hand_choose
#     ./young-hand
#     ./young-hand_rotate
#     ./dark-skin-hand
#     ./dark-skin-hand_rotate
#     ./light-skin-hand
#     ./light-skin-hand_rotate
# )
BLEND_DIRS=(
    ./aged-hand_paper
    # ./shorthairgirl
    # ./whitewoman
)

# HDR 文件所在目录
HDR_DIR=./hdrs3

# 遍历 HDR 目录下所有 .hdr 和 .exr 文件
for HDR in "$HDR_DIR"/*.{hdr,exr}; do
    # 检查文件是否存在（防止没有匹配到文件）
    [ -f "$HDR" ] || continue

    # 提取 HDR 文件名（不含路径和扩展名）
    HDR_NAME=$(basename "$HDR" | cut -d. -f1)
    echo "Processing HDR: $HDR_NAME"

    # 遍历所有 blend 目录
    for BLEND_DIR in "${BLEND_DIRS[@]}"; do
        for blend in "$BLEND_DIR"/*.blend; do
            [ -f "$blend" ] || continue
            echo "Rendering: $blend under HDR: $HDR_NAME"
            $BLENDER_PATH "$blend" --background --python "$PY_SCRIPT" -- "$blend" --hdr-name "$HDR_NAME"
        done
    done
done
