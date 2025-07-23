import cv2
import numpy as np
import os
from PyQt5.QtWidgets import QApplication, QMainWindow, QSlider, QLabel, QVBoxLayout, QWidget,QPushButton,QDesktopWidget
from PyQt5.QtCore import Qt
import sys
import tifffile
from scipy import stats
from skimage import io
import skimage

#找到当前根目录
current_directory = os.path.dirname(os.path.abspath(__file__))

def remove_outliers(png_path,radius=2, threshold=50, which_outliers='bright'):
    gray = cv2.imread(png_path, cv2.IMREAD_GRAYSCALE)

    # 4. 计算邻域均值
    kernel = np.ones((2 * radius + 1, 2 * radius + 1), np.float32) / ((2 * radius + 1) ** 2)
    mean_image = cv2.filter2D(gray, -1, kernel)

    # 5. 计算异常值
    if which_outliers == 'bright':
        outliers_mask = gray > (mean_image + threshold)
    elif which_outliers == 'dark':
        outliers_mask = gray < (mean_image - threshold)
    else:
        raise ValueError("Invalid 'which_outliers' value. Use 'bright' or 'dark'.")

    # 6. 去除异常值
    cleaned_image = np.where(outliers_mask, mean_image.astype(np.uint8), gray)

    return cleaned_image

def apply_processing():
    input_folder = current_directory + "\\val\\temp_png\\"
    output_folder = current_directory + "\\val\\traditional improvement\\remove_outliers.tif"
    # 获取所有PNG文件的文件列表
    png_files = [f for f in os.listdir(input_folder) if f.endswith('.png')]
    png_files.sort(key=lambda x: int(x.split("_")[-1].split(".")[0]))

    denoise_images=[]

    for png in png_files:
        png_path = os.path.join(input_folder, png)
        print(png_path)
        denoise_image=remove_outliers(png_path)
        denoise_images.append(denoise_image)
    denoise_images = np.array(denoise_images)
    denoise_images = (denoise_images.astype(np.uint16) << 8) | denoise_images.astype(np.uint16)
    tifffile.imwrite(output_folder, np.array(denoise_images))
    image_data = io.imread(output_folder)
    image_data = skimage.exposure.rescale_intensity(image_data, out_range=np.uint8)
    io.imsave(output_folder, image_data)
apply_processing()
