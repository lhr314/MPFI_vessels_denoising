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

def get_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--origin', action='store_true', help='原图')
    parser.add_argument('--AI_improve', action='store_true', help='AI')
    parser.add_argument('--AI_segmentation+denoise', action='store_true', help='降噪')
    opt = parser.parse_args()
    return opt

opt=get_opt();

if(opt.origin):
    images_folder="val/temp_png/"
    mode="origin"
elif(opt.AI_improve):
    images_folder="val/temp_segmentation/"
    mode = "AI_improve"
else:
    images_folder="val/temp_denoise/"
    mode = "AI_segmentation+denoise"

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

var_list=[]
for filename in os.listdir(images_folder):
    if filename.endswith(".png"):
        image_path = os.path.join(images_folder, filename)
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        #mean_frequency = calculate_high_frequency_mean_frequency(image)
        variance=wavelet(image)
        var_list.append(variance)
plt.plot(var_list, marker='',color='black')
if opt.origin:
    plt.title(f'Noise level of original images',fontsize=16)
elif opt.AI_improve:
    plt.title(f'Noise level of segmented images', fontsize=16)
else:
    plt.title(f'Noise level of segmented and denoise images', fontsize=16)
plt.xlabel('Image sequences',fontsize=16)
plt.ylabel('Noise level',fontsize=16)
plt.ylim(0,0.4)
plt.xlim(0,400)
plt.savefig(f'val/Noise_level_picture/{mode}/{mode}.png')
average=sum(var_list)/len(var_list)
print(f"{mode}平均噪音：{average}")