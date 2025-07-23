import cv2
import os
import numpy as np
import pywt
import matplotlib.pyplot as plt

def denoise_images_classtic(input_folder, output_folder, kernel_size=(5, 5), sigmaX=0):
    # 创建输出文件夹
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 遍历输入文件夹中的图像文件
    for filename in os.listdir(input_folder):
        if filename.endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            # 读取图像
            image_path = os.path.join(input_folder, filename)
            image = cv2.imread(image_path)
            # 应用高斯模糊进行降噪
            denoised_image = cv2.GaussianBlur(image, kernel_size, sigmaX)
            # 提高亮度
            denoised_brightened_image = cv2.convertScaleAbs(denoised_image, alpha=1.2, beta=0)
            # 形态学操作去除小噪点
            kernel = np.ones((5, 5), np.uint8)
            opening = cv2.morphologyEx(denoised_brightened_image, cv2.MORPH_OPEN, kernel)
            # 写入输出文件夹
            output_path = os.path.join(output_folder, filename)
            cv2.imwrite(output_path, opening)
            print(f"Denoised {filename} saved as {output_path}")

# 输入和输出文件夹
input_folder = "val/temp_noise"
output_folder = "val/temp_denoise"

# 高斯模糊参数
kernel_size = (5, 5)
sigmaX = 3.0


# 输入和输出文件夹
input_folder = "val/temp_noise"
output_folder = "val/temp_denoise"

# 执行图像降噪
denoise_images_classtic(input_folder, output_folder,kernel_size,sigmaX)