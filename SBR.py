import numpy as np
from scipy.optimize import curve_fit
import os
import cv2
import matplotlib.pyplot as plt
import math


def SBR(noise_img,clean_img):
    # Convert images to float32
    original_image = clean_img.astype(np.float32)
    noisy_image = noise_img.astype(np.float32)

    # Calculate the signal power
    signal_power = np.mean(original_image ** 2)

    # Calculate the noise power
    noise_power = np.mean((original_image - noisy_image) ** 2)

    # If noise power is zero, return infinity
    if noise_power == 0:
        return float('inf')

    # Calculate SNR
    snr = 10 * np.log10(signal_power / noise_power)
    return snr

def convert_png_to_SNR(input_folder, comparison_folder):
    # 获取所有PNG文件的文件列表
    png_files = [f for f in os.listdir(input_folder) if f.endswith('.png')]
    png_files.sort(key=lambda x: int(x.split("_")[-1].split(".")[0]))
    png_signal_files = [f for f in os.listdir(comparison_folder) if f.endswith('.png')]
    png_signal_files.sort(key=lambda x: int(x.split("_")[-1].split(".")[0]))
    # 用于存储图像数据的列表
    SBR_list = []

    # 将所有PNG图像粘贴到TIFF图像中
    for png_file,png_signal_file in zip(png_files,png_signal_files):
        png_path = os.path.join(input_folder, png_file)
        png_image = cv2.imread(png_path,cv2.IMREAD_GRAYSCALE)
        png_signal_path = os.path.join(comparison_folder, png_signal_file)
        png_signal_image = cv2.imread(png_signal_path, cv2.IMREAD_GRAYSCALE)
        # 应用LUT到目标TIFF图像
        SBR_list.append(SBR(png_image, png_signal_image))

    return SBR_list

original_images_folder="datasets/png/Original images/"
wavelet_denoising_folder="datasets/png/Wavelet denoising/"
Non_Local_Means_denoising_folder="datasets/png/Non-local Means denoising/"
Segmented_and_denoise_images_folder="datasets/png/AI improvement/"
Segmented_original_images_folder="datasets/png/Segmented_original_images/"

original_images_SBR=convert_png_to_SNR(original_images_folder,Segmented_original_images_folder)
wavelet_denoising_SBR=convert_png_to_SNR(wavelet_denoising_folder,Segmented_original_images_folder)
Non_Local_Means_denoising_SBR=convert_png_to_SNR(Non_Local_Means_denoising_folder,Segmented_original_images_folder)
Segmented_and_denoise_images_SBR=convert_png_to_SNR(Segmented_and_denoise_images_folder,Segmented_original_images_folder)

max_list=[max(original_images_SBR), max(wavelet_denoising_SBR), max(Non_Local_Means_denoising_SBR), max(Segmented_and_denoise_images_SBR)]
max_value=max(max_list)

plt.figure(dpi=300)
plt.plot(original_images_SBR,label='original images', color='gray',linewidth=2)
plt.plot(wavelet_denoising_SBR,label='Wavelet denoising', color='lightblue',linewidth=2)
plt.plot(Non_Local_Means_denoising_SBR,label='Non-Local Means denoising', color='yellow',linewidth=2)
plt.plot(Segmented_and_denoise_images_SBR,label='Segmented_and_denoise_images', color='orange',linewidth=2)
# 添加图例
plt.legend()
plt.xlabel('images sequence',fontweight='bold')
plt.ylabel('SNR(dB)',fontweight='bold')
plt.show()