import cv2
import numpy as np
import os
import tifffile
from PIL import Image, ImageOps
from skimage import exposure
from skimage import io
import skimage
# 创建CLAHE对象
def histogram_equalization(path):
    # 读取灰度图像
    image = cv2.imread(path,cv2.IMREAD_UNCHANGED)
    image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
    image = image.astype(np.uint8)
    # 3. 直方图均衡化
    equalized = cv2.equalizeHist(image)

    # 4. 计算饱和像素的阈值
    hist, bins = np.histogram(equalized.flatten(), 256, [0, 256])
    total_pixels = equalized.shape[0] * equalized.shape[1]
    cumulative_sum = np.cumsum(hist)
    threshold_index = np.argmax(cumulative_sum > total_pixels * 0.0035 / 100.0)
    threshold_value = bins[threshold_index]

    # 5. 调整对比度
    adjusted = np.clip((equalized - threshold_value) * 255.0 / (255.0 - threshold_value), 0, 255).astype(np.uint16)

    return adjusted

input_folder = "val/temp_png/"
output_folder = "val/traditional improvement/histogram_equalization.tif"
# 获取所有PNG文件的文件列表
png_files = [f for f in os.listdir(input_folder) if f.endswith('.png')]
png_files.sort(key=lambda x: int(x.split("_")[-1].split(".")[0]))

equalized_images=[]

for png in png_files:
    png_path = os.path.join(input_folder, png)

    equalized_image=histogram_equalization(png_path)
    equalized_images.append(equalized_image)

tifffile.imwrite(output_folder, np.array(equalized_images))
image_data = io.imread(output_folder)
image_data = skimage.exposure.rescale_intensity(image_data, out_range=np.uint8)
io.imsave(output_folder, image_data)

