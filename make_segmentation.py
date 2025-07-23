import torch
from glob import glob
import cv2
from tqdm import tqdm
import numpy as np
import os
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout, QProgressDialog,QProgressBar,QLabel,QDesktopWidget,QMessageBox
import sys
from PyQt5.QtCore import Qt, QThread, pyqtSignal

# 创建主窗口
app = QApplication(sys.argv)
window = QWidget()
# 设置窗口居中
window.setFixedSize(300, 100)
window_geometry = window.frameGeometry()
center_point = QDesktopWidget().availableGeometry().center()
window_geometry.moveCenter(center_point)
window.setGeometry(window_geometry)
window.setWindowTitle('正在生成分割图片中')
# 设置布局
layout = QVBoxLayout(window)
# 创建进度条和标签
progress_bar = QProgressBar()
png_files = len([f for f in os.listdir("val/temp_png") if f.lower().endswith('.png')])
progress_bar.setValue(0)
progress_bar.setRange(0,png_files)
progress_label = QLabel("进度")
layout.addWidget(progress_label)
layout.addWidget(progress_bar)
# 显示主窗口
window.show()
def format_number(num, length):
    return '{:0{}}'.format(num, len(str(length)))
def mask_parse(mask):
    mask = np.expand_dims(mask, axis=-1)  ## (512, 512, 1)
    mask = np.concatenate([mask, mask, mask], axis=-1)  ## (512, 512, 3)
    return mask
test_x = "val/temp_mask"
test_y = ("val/temp_png")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 获取所有PNG文件的文件列表
png_files_x = [f for f in os.listdir(test_x) if f.endswith('.png')]
png_files_x.sort(key=lambda a: int(a.split("mask")[-1].split(".")[0]))

png_files_y = [f for f in os.listdir(test_y) if f.endswith('.png')]
png_files_y.sort(key=lambda a: int(a.split("temp")[-1].split(".")[0]))

i=0

n= sum(1 for file in os.listdir("val/temp_mask") if file.lower().endswith('.png'))

for png_x, png_y in zip(png_files_x, png_files_y):
    # Reading image
    png_path_y = os.path.join(test_y, png_y)
    image = cv2.imread(png_path_y, cv2.IMREAD_COLOR)  ## (512, 512, 3)
    # Reading mask
    png_path_x = os.path.join(test_x, png_x)
    mask = cv2.imread(png_path_x, cv2.IMREAD_GRAYSCALE)  ## (512, 512)
    y = mask > 0.5
    y = np.array(y, dtype=np.uint8)
    # Saving masks
    y=mask_parse(y)

    cv2.imwrite(f"val/temp_segmentation/seg_{format_number(i, n)}.png", cv2.cvtColor(y*image, cv2.COLOR_BGR2GRAY))
    i+=1
    segment_file_count = sum(1 for file in os.listdir("val/temp_segmentation") if file.lower().endswith('.png'))
    progress_bar.setValue(segment_file_count)
    QApplication.processEvents()
    if progress_bar.value() == png_files:
        app.quit()