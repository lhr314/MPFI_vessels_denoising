import cv2
import numpy as np
import os
import tifffile
from PyQt5.QtWidgets import QApplication, QMainWindow, QSlider, QLabel, QVBoxLayout, QWidget,QPushButton,QDesktopWidget
from PyQt5.QtCore import Qt
import sys
from skimage import io
import skimage

#找到当前根目录
current_directory = os.path.dirname(os.path.abspath(__file__))

app = QApplication(sys.argv)
#创建参数选择窗口
window = QMainWindow()
window.setWindowTitle("Despeckle")
cp = QDesktopWidget().availableGeometry().center()
window.setGeometry(cp.x()-200, cp.y()-100, 400, 200)
# 创建布局
layout = QVBoxLayout()
# 创建Despeckle滑动条和标签
despeckle_label = QLabel('Despeckle Kernel Size: 5', window)
layout.addWidget(despeckle_label)
despeckle_slider = QSlider(Qt.Horizontal, window)
def Despeckle(path):
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    # 中值滤波去斑点
    kernel_size = despeckle_slider.value()
    despeckled_image = cv2.medianBlur(image, kernel_size)
    despeckled_image = despeckled_image.astype(np.uint16)
    return despeckled_image

def apply_processing():
    input_folder = current_directory + "\\val\\temp_png\\"
    output_folder = current_directory + "\\val\\traditional improvement\\Despeckled.tif"
    # 获取所有PNG文件的文件列表
    png_files = [f for f in os.listdir(input_folder) if f.endswith('.png')]
    png_files.sort(key=lambda x: int(x.split("_")[-1].split(".")[0]))

    despeckled_images=[]

    for png in png_files:
        png_path = os.path.join(input_folder, png)
        print(png_path)
        despeckled_image=Despeckle(png_path)
        despeckled_images.append(despeckled_image)
    despeckled_images = np.array(despeckled_images)
    despeckled_images = (despeckled_images.astype(np.uint16) << 8) | despeckled_images.astype(np.uint16)
    tifffile.imwrite(output_folder, np.array(despeckled_images))
    image_data = io.imread(output_folder)
    image_data = skimage.exposure.rescale_intensity(image_data, out_range=np.uint8)
    io.imsave(output_folder, image_data)
    window.close()

despeckle_slider.setMinimum(1)
despeckle_slider.setMaximum(15)
despeckle_slider.setValue(5)
despeckle_slider.setTickPosition(QSlider.TicksBelow)
despeckle_slider.setTickInterval(1)
despeckle_slider.valueChanged.connect(lambda value: despeckle_label.setText(f'Despeckle Kernel Size: {value}'))
layout.addWidget(despeckle_slider)
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