from PIL import Image
import numpy as np
import os
import tifffile
import cv2
import time
import argparse


def get_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--segmentation', action='store_true', help='分割')
    parser.add_argument('--denoise', action='store_true', help='降噪')
    opt = parser.parse_args()
    return opt

def convert_png_to_tif(input_folder, output_filename):
    # 获取所有PNG文件的文件列表
    png_files = [f for f in os.listdir(input_folder) if f.endswith('.png')]
    png_files.sort(key=lambda x: int(x.split("_")[-1].split(".")[0]))
    # 用于存储图像数据的列表
    image_data_list = []

    # 将所有PNG图像粘贴到TIFF图像中
    for png_file in png_files:
        png_path = os.path.join(input_folder, png_file)
        png_image = cv2.imread(png_path,cv2.IMREAD_GRAYSCALE)
        # 应用LUT到目标TIFF图像
        image_data_list.append(np.array(png_image))

    # 保存TIFF图像
    tif_path = output_filename
    tifffile.imwrite(tif_path, np.array(image_data_list))

    print(f'TIFF图像已成功保存为: {tif_path}')

# 设置目标TIFF图像路径和LUT文件路径
# 设置输入文件夹和输出文件名
opt=get_opt()
if opt.segmentation:
    input_folder_path = 'val/temp_segmentation'
    output_tif_filename = "val/AI_segmentation+denoise/seg/AI_segmentation.tif"
else:
    input_folder_path = 'val/temp_denoise'
    output_tif_filename = "val/AI_segmentation+denoise/denoise/AI_segmentation+denoise.tif"

# 调用函数进行转换
convert_png_to_tif(input_folder_path, output_tif_filename)


