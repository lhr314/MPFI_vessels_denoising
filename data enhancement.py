#数据增强
import os
from PIL import Image
def rotate_and_flip_images(input_folder, output_folder):
    # 遍历文件夹中的图像
    for filename in os.listdir(input_folder):
        if filename.endswith(".png") or filename.endswith(".jpg"):  # 假设你的图像是png或jpg格式
            input_path = os.path.join(input_folder, filename)

            # 打开图像
            image = Image.open(input_path)

            # 旋转和翻转操作
            rotations_and_flips = [
                ("rot90", image.rotate(90)),
                ("flip_vertical", image.transpose(Image.FLIP_TOP_BOTTOM)),
            ]
            i=2
            # 保存处理后的图像
            for suffix, transformed_image in rotations_and_flips:
                output_filename = f"{i}_{filename.split('.')[0]}_{suffix}.{filename.split('.')[1]}"
                output_path = os.path.join(output_folder, output_filename)
                transformed_image.save(output_path)
                i+=1

input_folder_A = "datasets/train/ori"
output_folder_A = "datasets/train/ori"  # 存储到原始文件夹中
input_folder_B = "datasets/train/mask"
output_folder_B = "datasets/train/mask"  # 存储到原始文件夹中


# 调用函数进行固定旋转和翻转
rotate_and_flip_images(input_folder_A, output_folder_A)
rotate_and_flip_images(input_folder_B, output_folder_B)