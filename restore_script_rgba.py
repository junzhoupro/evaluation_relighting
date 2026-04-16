#!/usr/bin/env python3
"""
复原脚本：将256x256的处理结果复原到原始512x512的位置和尺寸
支持RGBA图像处理
"""

import cv2
import numpy as np
from PIL import Image
import json
import argparse
import os

def restore_to_original(img_256_path, mask_256_path, metadata_path, original_img_path, output_dir=None):
    """
    将256x256的图像复原到原始512x512图片上
    
    Args:
        img_256_path: 256x256处理后的图片路径
        mask_256_path: 256x256的mask路径
        metadata_path: 元数据文件路径
        original_img_path: 原始512x512图片路径
        output_dir: 输出目录，如果不提供则使用当前目录
    
    Returns:
        restored_image: 复原后的图像
    """
    # 读取元数据
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    # 读取256x256的图像和mask
    img_256 = Image.open(img_256_path)
    mask_256 = Image.open(mask_256_path).convert('L')
    
    # 读取原始512x512图片作为背景
    if not os.path.exists(original_img_path):
        raise FileNotFoundError(f"找不到原始图片: {original_img_path}")
    
    original_img = Image.open(original_img_path)
    original_mode = metadata.get('original_mode', original_img.mode)
    
    print(f"原始图像模式: {original_mode}")
    print(f"256x256图像模式: {img_256.mode}")
    
    # 确保模式一致性
    if original_mode == 'RGBA':
        original_img = original_img.convert('RGBA')
        if img_256.mode != 'RGBA':
            img_256 = img_256.convert('RGBA')
    else:
        original_img = original_img.convert('RGB')
        if img_256.mode == 'RGBA':
            # 将RGBA转换为RGB（在白背景上合成）
            white_bg = Image.new('RGB', img_256.size, (255, 255, 255))
            img_256 = Image.alpha_composite(white_bg.convert('RGBA'), img_256).convert('RGB')
    
    if original_img.size != tuple(metadata['original_size']):
        print(f"警告: 原始图片尺寸 {original_img.size} 与元数据中的尺寸 {metadata['original_size']} 不匹配")
        original_img = original_img.resize(tuple(metadata['original_size']), Image.LANCZOS)
    
    # 获取原始尺寸和变换参数
    original_size = tuple(metadata['original_size'])
    crop_region = metadata['crop_region']
    
    print(f"原始尺寸: {original_size}")
    print(f"裁剪区域: {crop_region}")
    print(f"使用原始图片: {original_img_path}")
    
    # 使用原始图片作为背景
    background = original_img.copy()
    
    # 第1步: 将256x256放大到原始裁剪尺寸
    crop_size = crop_region['size']
    if crop_size > 0:
        img_scaled = img_256.resize((crop_size, crop_size), Image.LANCZOS)
        mask_scaled = mask_256.resize((crop_size, crop_size), Image.NEAREST)
    else:
        print("警告: 裁剪尺寸无效")
        return background
    
    # 第2步: 将放大后的图像放置到原始位置
    crop_x = crop_region['x']
    crop_y = crop_region['y']
    
    # 确保位置和尺寸在有效范围内
    crop_x = max(0, min(crop_x, original_size[0] - crop_size))
    crop_y = max(0, min(crop_y, original_size[1] - crop_size))
    
    # 调整裁剪尺寸以适应边界
    actual_width = min(crop_size, original_size[0] - crop_x)
    actual_height = min(crop_size, original_size[1] - crop_y)
    
    if actual_width != crop_size or actual_height != crop_size:
        img_scaled = img_scaled.crop((0, 0, actual_width, actual_height))
        mask_scaled = mask_scaled.crop((0, 0, actual_width, actual_height))
        print(f"调整尺寸: {actual_width}x{actual_height}")
    
    # 第3步: 合成图像
    bg_array = np.array(background)
    fg_array = np.array(img_scaled)
    mask_array = np.array(mask_scaled).astype(np.float32) / 255.0
    
    # 根据图像模式进行不同的合成处理
    if original_mode == 'RGBA':
        # 对于RGBA图像，只在mask区域更新像素
        if fg_array.shape[2] == 4:  # RGBA
            # 使用mask来控制哪些像素被更新
            for c in range(4):  # 包括alpha通道
                bg_array[crop_y:crop_y+actual_height, crop_x:crop_x+actual_width, c] = (
                    fg_array[:, :, c] * mask_array + 
                    bg_array[crop_y:crop_y+actual_height, crop_x:crop_x+actual_width, c] * (1 - mask_array)
                ).astype(np.uint8)
        else:  # RGB转RGBA
            for c in range(3):
                bg_array[crop_y:crop_y+actual_height, crop_x:crop_x+actual_width, c] = (
                    fg_array[:, :, c] * mask_array + 
                    bg_array[crop_y:crop_y+actual_height, crop_x:crop_x+actual_width, c] * (1 - mask_array)
                ).astype(np.uint8)
            # alpha通道按mask更新
            bg_array[crop_y:crop_y+actual_height, crop_x:crop_x+actual_width, 3] = (
                mask_array * 255 + 
                bg_array[crop_y:crop_y+actual_height, crop_x:crop_x+actual_width, 3] * (1 - mask_array)
            ).astype(np.uint8)
    else:
        # 对于RGB图像，在指定区域进行alpha混合
        for c in range(3):
            bg_array[crop_y:crop_y+actual_height, crop_x:crop_x+actual_width, c] = (
                fg_array[:, :, c] * mask_array + 
                bg_array[crop_y:crop_y+actual_height, crop_x:crop_x+actual_width, c] * (1 - mask_array)
            ).astype(np.uint8)
    
    # 创建复原后的图像
    if original_mode == 'RGBA':
        restored_image = Image.fromarray(bg_array, 'RGBA')
    else:
        restored_image = Image.fromarray(bg_array, 'RGB')
    
    # 同时创建复原后的mask
    restored_mask = Image.new('L', original_size, 0)
    if actual_width > 0 and actual_height > 0:
        restored_mask.paste(mask_scaled, (crop_x, crop_y))
    
    # 保存结果
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(img_256_path))[0]
        if base_name.endswith('_256'):
            base_name = base_name[:-4]  # 移除_256后缀
        
        # 根据原始模式保存文件
        if original_mode == 'RGBA':
            restored_img_path = os.path.join(output_dir, f"{base_name}_restored.png")
        else:
            restored_img_path = os.path.join(output_dir, f"{base_name}_restored.png")
        
        restored_mask_path = os.path.join(output_dir, f"{base_name}_mask_restored.png")
        
        restored_image.save(restored_img_path)
        restored_mask.save(restored_mask_path)
        
        print(f"保存复原图像: {restored_img_path}")
        print(f"保存复原mask: {restored_mask_path}")
    
    return restored_image, restored_mask

def restore_from_metadata_file(metadata_path, original_img_path, output_dir=None):
    """
    从元数据文件中读取路径信息并进行复原
    """
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    img_256_path = metadata['output_paths']['image']
    mask_256_path = metadata['output_paths']['mask']
    
    return restore_to_original(img_256_path, mask_256_path, metadata_path, original_img_path, output_dir)

def batch_restore(input_dir, original_imgs_dir, output_dir):
    """
    批量复原处理
    """
    metadata_files = [f for f in os.listdir(input_dir) if f.endswith('_metadata.json')]
    
    for metadata_file in metadata_files:
        metadata_path = os.path.join(input_dir, metadata_file)
        print(f"\n处理: {metadata_file}")
        
        # 根据元数据文件名推断原始图片名
        base_name = metadata_file.replace('_metadata.json', '')
        
        # 常见的图片扩展名
        extensions = ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG']
        original_img_path = None
        
        for ext in extensions:
            potential_path = os.path.join(original_imgs_dir, base_name + ext)
            if os.path.exists(potential_path):
                original_img_path = potential_path
                break
        
        if not original_img_path:
            print(f"找不到原始图片: {base_name}.*")
            continue
        
        try:
            restore_from_metadata_file(metadata_path, original_img_path, output_dir)
            print("复原成功!")
        except Exception as e:
            print(f"复原失败: {e}")

def main():
    parser = argparse.ArgumentParser(description='将256x256图像复原到原始512x512图片上')
    parser.add_argument('--mode', choices=['single', 'batch'], default='single', 
                       help='处理模式: single(单个文件) 或 batch(批量处理)')
    
    # 单个文件模式参数
    parser.add_argument('--image_256', type=str, help='256x256图片路径')
    parser.add_argument('--mask_256', type=str, help='256x256 mask路径')
    parser.add_argument('--metadata', type=str, help='元数据文件路径')
    parser.add_argument('--original', type=str, help='原始512x512图片路径')
    
    # 批量处理模式参数
    parser.add_argument('--input_dir', type=str, help='输入目录(包含元数据文件)')
    parser.add_argument('--original_dir', type=str, help='原始图片目录')
    
    # 通用参数
    parser.add_argument('--output_dir', type=str, default='./restored', help='输出目录')
    
    args = parser.parse_args()
    
    try:
        if args.mode == 'single':
            if not all([args.image_256, args.mask_256, args.metadata, args.original]):
                print("单个文件模式需要提供 --image_256, --mask_256, --metadata, --original 参数")
                return
            
            restored_img, restored_mask = restore_to_original(
                args.image_256, args.mask_256, args.metadata, 
                args.original, args.output_dir
            )
            print("复原完成!")
            
        elif args.mode == 'batch':
            if not all([args.input_dir, args.original_dir]):
                print("批量处理模式需要提供 --input_dir, --original_dir 参数")
                return
            
            batch_restore(args.input_dir, args.original_dir, args.output_dir)
            print("批量复原完成!")
            
    except Exception as e:
        print(f"处理失败: {e}")

if __name__ == "__main__":
    main()

# 使用示例:
# 单个文件复原到原始图片:
# python restore_to_512.py --mode single --image_256 output_256.png --mask_256 mask_256.png --metadata metadata.json --original original_512.png --output_dir ./restored

# 批量复原到原始图片:
# python restore_to_512.py --mode batch --input_dir ./output_256 --original_dir ./original_images --output_dir ./restored