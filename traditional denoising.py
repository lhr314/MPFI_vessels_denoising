import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import tifffile
import pywt
import argparse

def get_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--adaptive', action='store_true', help='非局部均值降噪')
    parser.add_argument('--wavelet', action='store_true', help='小波变换降噪')
    opt = parser.parse_args()
    return opt

opt=get_opt()
def adaptive_denoising(input_folder, output_filename):
    # 获取所有PNG文件的文件列表
    png_files = [f for f in os.listdir(input_folder) if f.endswith('.png')]
    png_files.sort(key=lambda x: int(x.split("_")[-1].split(".")[0]))
    # 用于存储图像数据的列表
    denoised_images = []
    # 将所有PNG图像粘贴到TIFF图像中
    for png_file in png_files:
        png_path = os.path.join(input_folder, png_file)
        png_image = cv2.imread(png_path, cv2.IMREAD_GRAYSCALE)
        # 计算局部方差
        local_var = cv2.filter2D(png_image, -1, np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]]))
        local_var = np.abs(local_var)
        # 根据局部方差计算阈值
        threshold = np.mean(local_var) + 2 * np.std(local_var)
        # 应用阈值处理
        denoised_image = cv2.threshold(png_image, threshold, 255, cv2.THRESH_TOZERO)[1]
        denoised_images.append(denoised_image)
    # 保存TIFF图像
    tif_path = output_filename
    tifffile.imwrite(tif_path, np.array(denoised_images))

def estimate_noise(coeff):
    """基于中值绝对偏差(MAD)估计噪声标准差"""
    if coeff.size == 0:
        return 0.0
    median_val = np.median(coeff)
    mad = np.median(np.abs(coeff - median_val))
    return mad / 0.6745  # 将MAD转换为标准差估计

def adaptive_wavelet_denoising(input_folder, output_filename, wavelet='db4', mode='sym', level=3):
    # 获取所有PNG文件的文件列表
    png_files = [f for f in os.listdir(input_folder) if f.endswith('.png')]
    png_files.sort(key=lambda x: int(x.split("_")[-1].split(".")[0]))

    denoised_images = []

    for png_file in png_files:
        png_path = os.path.join(input_folder, png_file)
        png_image = cv2.imread(png_path, cv2.IMREAD_GRAYSCALE)

        # 多级小波分解
        coeffs = pywt.wavedec2(png_image, wavelet=wavelet, level=level, mode=mode)

        # 对每层高频系数进行阈值处理
        for i in range(1, len(coeffs)):
            cH, cV, cD = coeffs[i]

            # 估计各方向噪声标准差
            sigma_cH = estimate_noise(cH)
            sigma_cV = estimate_noise(cV)
            sigma_cD = estimate_noise(cD)

            # 计算基于子带尺寸的自适应阈值（通用阈值公式）
            threshold_cH = sigma_cH * np.sqrt(2 * np.log(cH.size))
            threshold_cV = sigma_cV * np.sqrt(2 * np.log(cV.size))
            threshold_cD = sigma_cD * np.sqrt(2 * np.log(cD.size))

            # 应用软阈值处理
            cH = pywt.threshold(cH, threshold_cH, mode='soft')
            cV = pywt.threshold(cV, threshold_cV, mode='soft')
            cD = pywt.threshold(cD, threshold_cD, mode='soft')

            coeffs[i] = (cH, cV, cD)

        # 小波重构
        denoised_image = pywt.waverec2(coeffs, wavelet=wavelet, mode=mode)

        # 处理边界可能出现的尺寸差异
        denoised_image = denoised_image[:png_image.shape[0], :png_image.shape[1]]

        # 转换为8位图像
        denoised_image = np.clip(denoised_image, 0, 255).astype(np.uint8)
        denoised_images.append(denoised_image)

    # 保存TIFF图像
    tifffile.imwrite(output_filename, np.array(denoised_images))

# 调用函数进行转换
if(opt.wavelet==False):
    input_folder_path = 'val/temp_png'
    output_tif_filename = "val/traditional improvement/Non_local_means_improvement.tif"
    adaptive_denoising(input_folder_path, output_tif_filename)
else:
    input_folder_path = 'val/temp_png'
    output_tif_filename = "val/traditional improvement/wavelet_improvement.tif"
    adaptive_wavelet_denoising(input_folder_path, output_tif_filename)

