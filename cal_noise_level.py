import cv2
import numpy as np
from scipy.fftpack import fft2, fftshift
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import glob
import os
from shutil import copyfile
import matplotlib.pyplot as plt
import pywt
import argparse

def wavelet(image):
    # 小波变换
    coeffs = pywt.dwt2(image, 'haar')  # 使用 Haar 小波基进行二维离散小波变换

    # 获取小波系数
    cA, (cH, cV, cD) = coeffs  # cA: 近似系数，cH: 水平细节系数，cV: 垂直细节系数，cD: 对角线细节系数

    # 计算高频小波系数的能量与总能量的比例
    epsilon = 1e-10  # 很小的常数
    denominator = cA ** 2 + cH ** 2 + cV ** 2 + cD ** 2 + epsilon
    noise_intensity = (np.sum(cH ** 2) + np.sum(cV ** 2) + np.sum(cD ** 2)) / np.sum(denominator)
    return noise_intensity

images_folder='Noise_analysis/segmentation'
max_depth=1900
min_depth=1500

# 创建用于存储有效文件的列表
valid_files = []

# 遍历目录并筛选文件
for filename in os.listdir(images_folder):
    if filename.endswith(".png") and filename.startswith("Depth_"):
        print(filename)
        try:
            # 提取深度值
            depth_str = filename.split('_')[1].split('.')[0]
            depth = int(depth_str)*5
            print(f'depth={depth}')

            # 深度范围检查
            if min_depth <= depth <= max_depth:
                valid_files.append((depth, filename))
        except (IndexError, ValueError):
            # 处理不符合命名规范的文件
            continue

# 按深度值排序
valid_files.sort(key=lambda x: x[0])

var_list = []
for depth, filename in valid_files:
    image_path = os.path.join(images_folder, filename)
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if image is not None:
        variance = wavelet(image)
        var_list.append(variance)
    else:
        print(f"无法读取图像：{filename}")

# 绘图和统计保持不变
plt.plot(var_list, marker='', color='black')
plt.xlabel('Image sequences', fontsize=16)
plt.ylabel('Noise level', fontsize=16)
plt.ylim(0, 0.4)
plt.xlim(0, (max_depth-min_depth)/5)
plt.savefig(f'Noise_level.png',dpi=300)
average = sum(var_list) / len(var_list)
print(f"平均噪音：{average}")