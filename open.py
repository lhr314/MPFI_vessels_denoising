import cv2
import numpy as np
import os


source_file="val/temp_mask/"
destination_file="val/temp_mask/"


def Open(source_file,destination_file):
    # 获取所有PNG文件的文件列表
    png_files = [f for f in os.listdir(source_file) if f.endswith('.png')]
    png_files.sort(key=lambda x: int(x.split("_")[-1].split(".")[0]))
    for png_file in png_files:
        png_path = os.path.join(source_file, png_file)
        png_image = cv2.imread(png_path,cv2.THRESH_BINARY)
        kernel_size = (5,5)
        # 对图像进行开运算
        opened_image = cv2.morphologyEx(png_image, cv2.MORPH_OPEN, kernel_size)
        #opened_image = cv2.morphologyEx(opened_image, cv2.MORPH_CLOSE, kernel_size)
        out_path = os.path.join(destination_file, png_file)
        print(out_path)
        cv2.imwrite(out_path,opened_image)


Open(source_file,destination_file)