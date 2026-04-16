#!/bin/bash

# BLENDER_PATH=/snap/bin/blender
BLENDER_PATH=blender
PY_SCRIPT=render_default_eval.py
# BLEND_DIR=/home/jz927/Documents/relighting/composite_survey/composite-20250603T200521Z-1-001/composite/  # 放置 .blend 文件的目录
# BLEND_DIR=./aged-hand_choose_rotate
# BLEND_DIR=./blackgirl_necklaces
# BLEND_DIR=./whitegirl_necklaces
BLEND_DIR=./blackgirl_newnecklaces
# BLEND_DIR=./whitegirl_newnecklaces


for blend in "$BLEND_DIR"/*.blend; do
    echo "Rendering: $blend"
    $BLENDER_PATH "$blend" --background --python "$PY_SCRIPT" -- "$blend"
done

# BLEND_DIR=./aged-hand_rotate

# for blend in "$BLEND_DIR"/*.blend; do
#     echo "Rendering: $blend"
#     $BLENDER_PATH "$blend" --background --python "$PY_SCRIPT" -- "$blend"
# done

# BLEND_DIR=./young-hand

# for blend in "$BLEND_DIR"/*.blend; do
#     echo "Rendering: $blend"
#     $BLENDER_PATH "$blend" --background --python "$PY_SCRIPT" -- "$blend"
# done

# BLEND_DIR=./young-hand_rotate

# for blend in "$BLEND_DIR"/*.blend; do
#     echo "Rendering: $blend"
#     $BLENDER_PATH "$blend" --background --python "$PY_SCRIPT" -- "$blend"
# done

# BLEND_DIR=./dark-skin-hand

# for blend in "$BLEND_DIR"/*.blend; do
#     echo "Rendering: $blend"
#     $BLENDER_PATH "$blend" --background --python "$PY_SCRIPT" -- "$blend"
# done

# BLEND_DIR=./dark-skin-hand_rotate

# for blend in "$BLEND_DIR"/*.blend; do
#     echo "Rendering: $blend"
#     $BLENDER_PATH "$blend" --background --python "$PY_SCRIPT" -- "$blend"
# done

# BLEND_DIR=./light-skin-hand

# for blend in "$BLEND_DIR"/*.blend; do
#     echo "Rendering: $blend"
#     $BLENDER_PATH "$blend" --background --python "$PY_SCRIPT" -- "$blend"
# done

# BLEND_DIR=./light-skin-hand_rotate

# for blend in "$BLEND_DIR"/*.blend; do
#     echo "Rendering: $blend"
#     $BLENDER_PATH "$blend" --background --python "$PY_SCRIPT" -- "$blend"
# done
