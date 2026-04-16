#!/bin/bash

# python extract_crop_rgba.py \
# --image aged-hand_choose/ballroom_4k/aged-hand_amanda-ring-v3-red_gold/aged-hand_amanda-ring-v3-red_gold.png \
# --mask aged-hand_choose/ballroom_4k/aged-hand_amanda-ring-v3-red_gold/mask/aged-hand_amanda-ring-v3-red_gold_mask_1.png \
# --output_dir ./output

# BLENDER_PATH=/snap/bin/blender
# PY_SCRIPT=render_under_hdr.py

# 所有 blend 文件所在目录（可以一次处理多个目录）
BLEND_DIRS=(
    # ./aged-hand_choose
    # ./aged-hand_choose_rotate
    # ./young-hand
    # ./young-hand_rotate
    # ./dark-skin-hand
    # ./dark-skin-hand_rotate
    # ./light-skin-hand
    # ./light-skin-hand_rotate
)
# BLEND_DIRS=(
#     ./blackgirl
#     ./shorthairgirl
#     ./whitewoman
# )
BLEND_DIRS=(
    ./blackgirl_newnecklaces
    ./whitegirl_newnecklaces
)

# 遍历所有 blend 目录
for BLEND_DIR in "${BLEND_DIRS[@]}"; do
    defaultdir="${BLEND_DIR}/render_default"
    blend_name=$(basename "$BLEND_DIR")   # aged-hand_choose 这样的名字


    # 遍历 defaultdir 下的子目录
    for jewelry_dir in "${defaultdir}"/*; do
        if [ -d "$jewelry_dir" ]; then
            filename=$(basename "$jewelry_dir")

            python extract_crop_rgba.py \
                --image "${jewelry_dir}/${filename}.png" \
                --mask "${jewelry_dir}/mask/${filename}_mask_1.png" \
                --output_dir "/home/jz927/Documents/relighting/neuralgaffer/Neural_Gaffer/preprocessed_data_${blend_name}"
        fi
    done
done

