#!/bin/bash

BLENDER_PATH=/snap/bin/blender
PY_SCRIPT=render_default_eval_rotate.py
# BLEND_DIR=/home/jz927/Documents/relighting/composite_survey/composite-20250603T200521Z-1-001/composite/  # 放置 .blend 文件的目录
BLEND_DIR=./dark-skin-hand

for blend in "$BLEND_DIR"/*.blend; do
    echo "Rendering: $blend"
    $BLENDER_PATH "$blend" --background --python "$PY_SCRIPT" -- "$blend"
done
