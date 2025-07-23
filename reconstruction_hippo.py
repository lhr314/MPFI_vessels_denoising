from PIL import Image
import numpy as np
import os

# 设置工作目录（替换为你的实际文件夹路径）
folder_path = 'datasets/tiff/hippocampus'

# 获取16位图像的尺寸和模式（使用一张存在的图片作为参考）
sample_image = None
for i in range(516):
    file_name = f'ICG_hippocampus_5160{i:03d}.png'
    file_path = os.path.join(folder_path, file_name)
    if os.path.exists(file_path):
        sample_image = Image.open(file_path)
        break

if sample_image is None:
    raise FileNotFoundError("未找到任何图像文件，请检查文件夹路径和文件命名")

# 获取图像尺寸和模式
width, height = sample_image.size
image_mode = sample_image.mode
bit_depth = sample_image.getbands()[0].bit_depth if hasattr(sample_image.getbands()[0], 'bit_depth') else 8
sample_image.close()

print(f"图像尺寸: {width}x{height}, 模式: {image_mode}, 位深度: {bit_depth}位")

# 创建16位全黑图像
if image_mode == 'I;16':  # 16位灰度图像
    # 创建16位无符号整型的全零数组
    black_array = np.zeros((height, width), dtype=np.uint16)
    black_image = Image.fromarray(black_array, mode='I;16')

elif image_mode == 'RGB' or image_mode == 'RGB;16':
    # 创建16位RGB全黑图像
    black_array = np.zeros((height, width, 3), dtype=np.uint16)
    black_image = Image.fromarray(black_array, mode='RGB')

elif image_mode == 'RGBA' or image_mode == 'RGBA;16':
    # 创建16位RGBA全黑图像（带透明通道）
    black_array = np.zeros((height, width, 4), dtype=np.uint16)
    black_image = Image.fromarray(black_array, mode='RGBA')

else:
    # 对于其他16位模式，使用通用方法
    print(f"警告: 检测到非常规16位图像模式 '{image_mode}'，尝试通用方法")
    try:
        # 创建适合该模式的16位零数组
        channels = len(image_mode)
        black_array = np.zeros((height, width, channels), dtype=np.uint16)
        black_image = Image.fromarray(black_array, mode=image_mode)
    except Exception as e:
        print(f"创建全黑图像失败: {e}")
        # 回退到16位灰度模式
        black_array = np.zeros((height, width), dtype=np.uint16)
        black_image = Image.fromarray(black_array, mode='I;16')

# 处理0-239和300-515范围的图片
processed_count = 0
for i in range(0, 240):  # 0-239
    file_name = f'ICG_hippocampus_5160{i:03d}.png'
    file_path = os.path.join(folder_path, file_name)
    if os.path.exists(file_path):
        # 保存为16位PNG，确保保留位深度
        black_image.save(file_path, format='PNG', bitdepth=16)
        processed_count += 1
        print(f"已处理: {file_name}")

for i in range(300, 516):  # 300-515
    file_name = f'ICG_hippocampus_5160{i:03d}.png'
    file_path = os.path.join(folder_path, file_name)
    if os.path.exists(file_path):
        black_image.save(file_path, format='PNG', bitdepth=16)
        processed_count += 1
        print(f"已处理: {file_name}")

print(f"\n操作完成！共处理 {processed_count} 个文件")
print(f"保留的图片范围: 240-299 (共60张图片)")
print(f"生成的全黑图像格式: {image_mode}, 16位深度")