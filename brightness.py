import cv2
import numpy as np
import os
import tifffile
from PyQt5.QtWidgets import QApplication, QMainWindow, QSlider, QLabel, QVBoxLayout, QWidget,QPushButton,QDesktopWidget
from PyQt5.QtCore import Qt
import sys
from PIL import Image, ImageEnhance



#找到当前根目录
current_directory = os.path.dirname(os.path.abspath(__file__))

app = QApplication(sys.argv)
#创建参数选择窗口
window = QMainWindow()
window.setWindowTitle("Brightness")
cp = QDesktopWidget().availableGeometry().center()

window.setGeometry(cp.x()-200, cp.y()-100, 400, 200)
# 创建布局
layout = QVBoxLayout()
# 创建Despeckle滑动条和标签
brightness_label = QLabel('Brightness: 1.0', window)
layout.addWidget(brightness_label)
brightness_slider = QSlider(Qt.Horizontal, window)

# 亮度调整函数
def adjust_brightness(image, factor):
    # 将图像转换为浮点数类型
    float_image = image.astype(np.float32)

    # 调整亮度
    adjusted_image = float_image * factor

    # 将调整后的图像限制在0-255之间，并转换为整数类型
    adjusted_image = np.clip(adjusted_image, 0, 255).astype(np.uint8)
    return adjusted_image
def brightness(path):
    image = Image.open(path)
    factor = brightness_slider.value()/10.0
    enhancer = ImageEnhance.Brightness(image)
    enhanced_image = enhancer.enhance(factor)
    return enhanced_image

def apply_processing():
    input_folder = current_directory + "\\val\\temp_png\\"
    output_folder = current_directory + "\\val\\traditional improvement\\Brightness.tif"
    # 获取所有PNG文件的文件列表
    png_files = [f for f in os.listdir(input_folder) if f.endswith('.png')]
    png_files.sort(key=lambda x: int(x.split("_")[-1].split(".")[0]))

    images=[]

    for png in png_files:
        png_path = os.path.join(input_folder, png)
        print(png_path)
        image=brightness(png_path)
        images.append(image)
    images = np.array(images)
    images_16bit = (images.astype(np.uint16) << 8) | images.astype(np.uint16)
    tifffile.imwrite(output_folder, np.array(images_16bit))
    window.close()

brightness_slider.setMinimum(1)
brightness_slider.setMaximum(20)
brightness_slider.setValue(10)
brightness_slider.setTickPosition(QSlider.TicksBelow)
brightness_slider.setTickInterval(1)
brightness_slider.valueChanged.connect(lambda value: brightness_label.setText(f'Brightness: {value/10.0}'))
layout.addWidget(brightness_slider)
# 创建Apply按钮
apply_button = QPushButton('Apply', window)
apply_button.clicked.connect(apply_processing)
layout.addWidget(apply_button)
# 设置中心窗口
central_widget = QWidget()
central_widget.setLayout(layout)
window.setCentralWidget(central_widget)

window.show()
sys.exit(app.exec_())