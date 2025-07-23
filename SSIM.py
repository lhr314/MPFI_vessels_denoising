import numpy as np
import cv2
from skimage.metrics import structural_similarity as ssim
import os
import matplotlib.pyplot as plt

def calculate_ssim(img1, img2):
    # 计算SSIM
    ssim_score, _ = ssim(img1, img2, full=True)
    return ssim_score

# Load dataset
img1 = "datasets/png"
img2 = "datasets/png"

# 获取所有PNG文件的文件列表
png_files_y1 = [f for f in os.listdir(img1) if f.endswith('.png')]
png_files_y1.sort(key=lambda a: int(a.split("_")[-1].split(".")[0]))
#png_files_y1.sort(key=lambda a: int(a.split("_")[-1].split(".")[0].split("mask")[1]))
png_files_y2 = [f for f in os.listdir(img2) if f.endswith('.png')]
png_files_y2.sort(key=lambda a: int(a.split("_")[-1].split(".")[0]))
#png_files_y2.sort(key=lambda a: int(a.split("_")[-1].split(".")[0].split("segmentation")[1]))
#png_files_y2.sort(key=lambda a: int(a.split("_")[-1].split(".")[0].split("mask")[1]))

SSIM_list = []
for png_x, png_y in zip(png_files_y1, png_files_y2):
    png_path_x = os.path.join(img1, png_x)
    image1 = cv2.imread(png_path_x, cv2.IMREAD_GRAYSCALE)
    png_path_y = os.path.join(img2, png_y)
    image2 = cv2.imread(png_path_y, cv2.IMREAD_GRAYSCALE)
    SSIM_list.append(calculate_ssim(image1, image2))
    print(SSIM_list[-1])
print(len(SSIM_list))
plt.plot(SSIM_list, marker='',color='black')
plt.title(f'SSIM_list',fontsize=16)
plt.xlabel('Image sequences',fontsize=16)
plt.ylabel('SSIM',fontsize=16)
plt.ylim(-1,1)
plt.xlim(0,400)
plt.show()
plt.savefig(f'SSIM index/SSIM_List.png')