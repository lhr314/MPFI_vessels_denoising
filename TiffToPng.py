import os
import cv2
import tifffile
import numpy as np
from PIL import Image
import argparse

def get_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', action='store_true', help='测试模式')
    parser.add_argument('--denoise', action='store_true', help='降噪模式')
    opt = parser.parse_args()
    return opt


def is_all_black_frame(image):
    # 获取图像的像素数据
    pixels = image.flatten()

    # 判断是否所有像素都是黑色 (0)
    return all(pixel == 0 for pixel in pixels)

#输出格式
def format_number(num, length):
    return '{:0{}}'.format(num, len(str(length)))

# def tiff_to_png(tiff_file_path, output_folder):
#     # 读取tif文件
#     tiff_data = tifffile.imread(tiff_file_path)
#     file_name = tiff_file_path.split("/")[-1].split(".")[0]
#     # 确保输出文件夹存在
#     os.makedirs(output_folder, exist_ok=True)
#
#     # 获取深度图像序列的数量
#     num_frames = tiff_data.shape[0]
#
#     # 循环遍历每一帧并保存为png
#     for i in range(num_frames):
#         # Get the current depth image
#         current_frame = tiff_data[i]
#
#         # Adjust the depth image range
#         adjusted_frame = cv2.normalize(current_frame, None, 0, 65535, cv2.NORM_MINMAX)
#
#         # Convert to 16-bit unsigned integer type
#         adjusted_frame = adjusted_frame.astype('uint16')
#
#         # Convert to 8-bit (grayscale)
#         adjusted_frame_8bit = cv2.convertScaleAbs(adjusted_frame, alpha=(255.0 / 65535.0))
#
#         # Generate output file path
#         output_path = os.path.join(output_folder, f"{file_name}_depth_{format_number(i+1, num_frames)}.png")
#         # Save as PNG file
#         if opt.test:
#             cv2.imwrite(output_path, adjusted_frame_8bit)
#         elif opt.denoise:
#             cv2.imwrite(output_path, current_frame)
#         else:
#             cv2.imwrite(output_path, adjusted_frame_8bit)
#         print(f"Frame {i + 1} saved as {output_path}")

def tiff_to_png(tiff_file_path, output_folder):
    tiff_image = Image.open(tiff_file_path)
    file_name = tiff_file_path.split("/")[-1].split(".")[0]
    num_frames = tiff_image.n_frames
    frame = 0
    try:
        while True:
            # 获取当前帧
            tiff_image.seek(frame)
            frame_image = tiff_image.convert('L')
            # 保存为png格式
            frame_path = os.path.join(output_folder, f"{file_name}_depth_{format_number(frame+1, num_frames)}.png")
            frame_image.save(frame_path, format='PNG')

            print(f'Saved {frame_path}')
            frame += 1

    except EOFError:
        # 当到达文件结尾时，会抛出EOFError
        print('All frames have been saved.')


def convert_tiff_folder(input_folder, output_folder):
    # 确保输出文件夹存在
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 获取输入文件夹中所有的tif文件
    tiff_files = [f for f in os.listdir(input_folder) if f.endswith('.tif') or f.endswith('.tiff')]

    for tiff_file in tiff_files:
        input_tiff_path = os.path.join(input_folder, tiff_file)
        tiff_to_png(input_tiff_path, output_folder)

opt = get_opt()
input_folder_path = "./val/"
output_folder_path = "./val/temp_png/"
convert_tiff_folder(input_folder_path, output_folder_path)