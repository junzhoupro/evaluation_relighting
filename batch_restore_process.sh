# hand_name=aged-hand_choose
# hand_name=aged-hand_choose_rotate
# hand_name=young-hand
# hand_name=young-hand_rotate
# hand_name=light-skin-hand
# hand_name=light-skin-hand_rotate
# hand_name=dark-skin-hand
# hand_name=dark-skin-hand_rotate
# hand_name=blackgirl
# hand_name=shorthairgirl
# hand_name=whitewoman
# hand_name=blackgirl_newnecklaces
hand_name=whitegirl_newnecklaces
REAL_DATA=/home/jz927/Documents/relighting/neuralgaffer/Neural_Gaffer/real_data_relighting_$hand_name/
HDR_DIR=/home/jz927/Documents/riggedhands/rigged-hand/hdrs_chosen

# 遍历所有 .exr 文件
for hdr_path in "$HDR_DIR"/*.exr; do
    # 获取不带后缀的 hdr 名称
    hdr_name=$(basename "$hdr_path" .exr)
    echo "当前 HDR: $hdr_name"

    # 遍历 real_data 下所有子目录
    for dir in "$REAL_DATA"*/; do
        if [ -d "$dir" ]; then
            dirname=$(basename "$dir")
            clean_name=${dirname%_256}

            echo "原目录名: $dirname"
            echo "处理后: $clean_name"

            img_dir="$REAL_DATA$dirname/pred_image/${hdr_name}_059.png"

            mask_dir="/home/jz927/Documents/relighting/neuralgaffer/Neural_Gaffer/preprocessed_data_${hand_name}/mask/"
            mask_name="$mask_dir$dirname.png"
            echo "mask: $mask_name"

            meta_dir="/home/jz927/Documents/relighting/neuralgaffer/Neural_Gaffer/preprocessed_data_${hand_name}/${clean_name}_metadata.json"
            echo "meta dir: $meta_dir"

            original_dir="/home/jz927/Documents/riggedhands/rigged-hand/${hand_name}/${hdr_name}/${clean_name}/${clean_name}.png"
            echo "original_dir: $original_dir"

            restore_dir="./restore2/${hand_name}/$hdr_name/$clean_name"
            mkdir -p "$restore_dir"

            python3 restore_script_rgba.py \
                --mode single \
                --image_256 "$img_dir" \
                --mask_256 "$mask_name" \
                --metadata "$meta_dir" \
                --original "$original_dir" \
                --output_dir "$restore_dir"

            echo "---"
        fi
    done
done
