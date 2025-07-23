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
    parser.add_argument('--test', action='store_true', help='测试模式')
    opt = parser.parse_args()
    return opt


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

def calculate_variance_threshold(images_folder):
    # 收集所有图片分割区域的平均像素方差
    coef_list=[]
    for filename in os.listdir(images_folder):
        if filename.endswith(".png"):
            image_path = os.path.join(images_folder, filename)
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            noise_level=wavelet(image)
            print(noise_level)
            coef_list.append(noise_level)
    plt.plot(coef_list, marker='', color='black')
    plt.title('Noise level')
    plt.xlabel('Image sequences')
    plt.ylabel('var')
    plt.ylim(0,0.4)
    plt.xlim(0, 400)
    if(opt.test==False):
        plt.show()
    # 尝试不同的阈值并计算类间方差
    max_variance = 0
    best_threshold = 0
    for threshold in np.linspace(min(coef_list), max(coef_list), 5000):
        below_threshold = [f for f in coef_list if f <= threshold]
        above_threshold = [f for f in coef_list if f > threshold]

        if len(below_threshold) == 0 or len(above_threshold) == 0:
            continue

        variance = len(below_threshold) * len(above_threshold) * (
                    np.mean(below_threshold) - np.mean(above_threshold)) ** 2

        if variance > max_variance:
            max_variance = variance
            best_threshold = threshold

    return best_threshold



def separate_images(input_folder, output_folder_noise, output_folder_noiseless, threshold,IsFirst):
    for filename in os.listdir(input_folder):
        if filename.endswith(".png"):
            image_path = os.path.join(input_folder, filename)
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            noise_level=wavelet(image)

            if noise_level > threshold:
                output_path = os.path.join(output_folder_noise, filename)
                if IsFirst==False:
                    os.remove(image_path)
                print(f"noise:{filename}")
            else:
                output_path = os.path.join(output_folder_noiseless, filename)
                print(f"noiseless:{filename}")
            cv2.imwrite(output_path, image)


if __name__ == "__main__":
    opt=get_opt()
    if(opt.test):
        input_folder = "val/temp_png/"
        output_folder_noise = "val/temp_noise/"
        output_folder_noiseless = "val/temp_denoise/"
    else:
        input_folder = "datasets/segmentation_png/"
        output_folder_noise = "datasets/Noise Separation/real noise"
        output_folder_noiseless = "datasets/Noise Separation/real noiseless"
    # 动态计算阈值
    threshold = calculate_variance_threshold(input_folder)
    print(f"First Calculated Threshold: {threshold}")
    # 阈值分离
    # 第一次分离
    separate_images(input_folder, output_folder_noise, output_folder_noiseless, threshold,True)
    # 第二次分离
    threshold = calculate_variance_threshold(output_folder_noiseless)
    print(f"Second Calculated Threshold: {threshold}")
    separate_images(output_folder_noiseless, output_folder_noise, output_folder_noiseless, threshold, False)
    number_noise=sum(1 for file in os.listdir("val/temp_noise") if file.lower().endswith('.png'))
    print(f"number of noise images: {number_noise}")
    #收集所有图片的高频平均频率
    coef_list = []
    for filename in os.listdir(output_folder_noiseless):
        if filename.endswith(".png"):
            image_path = os.path.join(output_folder_noiseless, filename)
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            noise_level = wavelet(image)
            print(noise_level)
            coef_list.append(noise_level)
    plt.plot(coef_list, marker='', color='black')
    plt.title('Noise level')
    plt.xlabel('Image sequences')
    plt.ylabel('var')
    plt.ylim(0, 0.4)
    plt.xlim(0,400)
    if (opt.test == False):
        plt.show()