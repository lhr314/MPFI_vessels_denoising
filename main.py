import napari
import torch
from skimage import io
import subprocess
from PyQt5.QtWidgets import QLabel, QVBoxLayout, QWidget, QLineEdit,QFormLayout,QPushButton,QMessageBox,QDesktopWidget
from PyQt5.QtWidgets import QMenuBar, QFileDialog, QHBoxLayout
from PyQt5.QtGui import QPixmap
import os
from tifffile import imwrite
import skimage
import numpy as np


#找到当前根目录
current_directory = os.path.dirname(os.path.abspath(__file__))

# 创建napari查看器
viewer = napari.Viewer(title="blood vessel 3D segmentation and denosing")

# 删除napari原有的默认主菜单
main_window = viewer.window.qt_viewer.window()
empty_menu_bar = QMenuBar()
main_window.setMenuBar(empty_menu_bar)
z_axis_spacing = 7.53

# 删除napari原有的窗口
try:
    from napari._qt.widgets.qt_welcome import QtWelcomeWidget
    welcome_widget = viewer.window.qt_viewer.findChild(QtWelcomeWidget)
    if welcome_widget:
        welcome_widget.hide()
except ImportError:
    print("QtWelcomeWidget import failed. Please check napari version.")

# 创建基于PyQT5的功能窗口
widget = QWidget()
layout = QVBoxLayout()

# 调整z轴间距
def update_z_height(value):
    z_axis_spacing = value
    layers = viewer.layers
    for layer in layers:
        layer.scale = (z_axis_spacing, 1, 1)
# 添加输入用于调整Z轴层间距
z_height_label = QLabel("Z-axis Slice Spacing (pixel)")
z_height_Edit = QLineEdit()
z_height_Edit.setFixedWidth(100)
z_height_Edit.textChanged.connect(update_z_height)

# 添加输入用输入tif文件
file_label = QLabel("Import TIF file")
file_open = QPushButton('Open')

#用户打开并输入tif文件
def open_tif_file():
    global image
    options = QFileDialog.Options()
    options |= QFileDialog.ReadOnly
    file_name, _ = QFileDialog.getOpenFileName(None, "Open TIF file", "", "TIFF Files (*.tif);;All Files (*)", options=options)
    if file_name:
        image_data = io.imread(file_name)
        image_data = skimage.exposure.rescale_intensity(image_data, out_range=np.uint8)
        io.imsave(current_directory+"\\val\\temp.tif",image_data)
        image=viewer.add_image(image_data, scale=(z_axis_spacing, 1, 1), colormap='red')
file_open.clicked.connect(open_tif_file)

#将选中图层保存为tif文件
Save_label = QLabel("Save TIF file:")
SaveTif = QPushButton('Save')
def save_button_callback(viewer, save_path):
    selected_layers = viewer.layers.selection
    selected_layers.update()
    for layer in selected_layers:
        napari.save_layers(save_path, [layer])
def on_save_button_click(viewer):
    save_path, _ = QFileDialog.getSaveFileName(caption="Save Tiff", filter="Tiff files (*.tif *.tiff)")
    if save_path:
        save_button_callback(viewer, save_path)
SaveTif.clicked.connect(lambda: on_save_button_click(viewer))

#保存当前用户打开的视图
Save_snapshot_label = QLabel("Save current view:")
Save_snapshot = QPushButton('Save snapshot')
def save_tiff(image, save_path):
    imwrite(save_path, image)
def save_view():
    # 使用napari内置的保存快照功能
    filepath, _ = QFileDialog.getSaveFileName(None,'Save Image', '', 'PNG Files (*.png);;JPEG Files (*.jpeg)')
    if filepath:
        viewer.screenshot(filepath)
Save_snapshot.clicked.connect(save_view)


# 添加调整背景颜色的按键
background_label=QLabel("Background adjustment:")
black_background = QPushButton("Black")
black_background.setFixedWidth(150)
white_background = QPushButton("White")
white_background.setFixedWidth(150)
# 调整背景颜色
def changeBackgroundToBlack():
    viewer.window.qt_viewer.canvas.background_color_override='black'
    viewer.window.qt_viewer.canvas.bgcolor='black'
def changeBackgroundToWhite():
    viewer.window.qt_viewer.canvas.background_color_override='white'
    viewer.window.qt_viewer.canvas.bgcolor = 'white'

black_background.clicked.connect(changeBackgroundToBlack)
white_background.clicked.connect(changeBackgroundToWhite)

#清理暂存文件函数
def delete_files_in_folder(folder_path):
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        try:
            if os.path.isfile(file_path):
                os.remove(file_path)
            elif os.path.isdir(file_path):
                # 如果要删除文件夹中的子文件夹及其内容，可以使用递归
                delete_files_in_folder(file_path)
        except Exception as e:
            print(f"Error deleting {file_path}: {e}")

# 添加按钮，通过预训练的模型优化当前打开的原始图像，并显示
AI_segmentation = QPushButton('Deep learning vessel segmentation')
def run_AI_segmentation():
    selected_layers = viewer.layers.selection
    for layer in selected_layers:
        napari.save_layers(current_directory + '\\val\\temp.tif', [layer])
    script_path1 = current_directory + '\\TiffToPng.py'
    script_path2 = current_directory + '\\test.py'
    script_path3= current_directory + '\\PngToTiff.py'
    if len(viewer.layers.selection) == 0:
        window = QWidget()
        cp = QDesktopWidget().availableGeometry().center()
        window.setGeometry(cp.x() - 200, cp.y() - 100, 400, 200)
        QMessageBox.warning(window, 'Warning', 'Please select a layer or import an image first')
    else:
        try:
            subprocess.run(['python', script_path1,'--test'])
            if torch.cuda.is_available():
                #subprocess.run(['python', script_path2, '--cuda', '--U_net'])
                #subprocess.run(['python', script_path2, '--cuda', '--NestedU_net'])
                # subprocess.run(['python', script_path2, '--cuda', '--FCN_Resnet'])
                #subprocess.run(['python', script_path2, '--cuda', '--DeeplabV3'])
                subprocess.run(['python', script_path2, '--cuda', '--ResU_net'])
                #subprocess.run(['python', script_path2,'--cuda','--ResU_net_Transformer'])
            else:
                subprocess.run(['python', script_path2, '--ResU_net_Transformer'])
            subprocess.run(['python', script_path3,'--segmentation'])
        except Exception as e:
            print(f'执行脚本时发生错误：{e}')
        image_data = io.imread(current_directory+'\\val\\AI_segmentation+denoise/seg/AI_segmentation.tif')
        delete_files_in_folder(current_directory+'\\val\\AI_segmentation+denoise/seg')
        delete_files_in_folder(current_directory+"\\val\\temp_png")
        delete_files_in_folder(current_directory+"\\val\\temp_mask")
        delete_files_in_folder(current_directory+"\\val\\temp_segmentation")
        viewer.add_image(image_data, scale=(5, 1, 1), colormap='red')
AI_segmentation.clicked.connect(run_AI_segmentation)

denoising = QPushButton('Post-process')
def run_denoising():
    selected_layers = viewer.layers.selection
    for layer in selected_layers:
        napari.save_layers(current_directory + '\\val\\temp.tif', [layer])
    script_path1 = current_directory + '\\TiffToPng.py'
    script_path2 = current_directory + '\\noise_separation.py'
    script_path3 = current_directory + '\\denoising.py'
    script_path5 = current_directory + '\\PngToTiff.py'
    if len(viewer.layers.selection) == 0:
        window = QWidget()
        cp = QDesktopWidget().availableGeometry().center()
        window.setGeometry(cp.x() - 200, cp.y() - 100, 400, 200)
        QMessageBox.warning(window, 'Warning', 'Please select a layer or import an image first')
    else:
        try:
            subprocess.run(['python', script_path1, '--denoise'])
            subprocess.run(['python', script_path2, '--test'])
            subprocess.run(['python', script_path3])
            subprocess.run(['python', script_path5,'--denoise'])
        except Exception as e:
            print(f'执行脚本时发生错误：{e}')
        image_data = io.imread(current_directory + '\\val\\AI_segmentation+denoise\\denoise\\AI_segmentation+denoise.tif')
        viewer.add_image(image_data, scale=(5, 1, 1), colormap='red')
        delete_files_in_folder(current_directory + '\\val\\AI_segmentation+denoise\\denoise')
        delete_files_in_folder(current_directory + "\\val\\temp_denoise")
        delete_files_in_folder(current_directory + "\\val\\temp_noise")
        delete_files_in_folder(current_directory + "\\val\\temp_png")
denoising.clicked.connect(run_denoising)

#局部方差自适应阈值降噪算法
adaptive_denoising = QPushButton('ATLV')
def run_adaptive_denoising():
    selected_layers = viewer.layers.selection
    for layer in selected_layers:
        napari.save_layers(current_directory + '\\val\\temp.tif', [layer])
    script_path1 = current_directory + '\\TiffToPng.py'
    script_path2 = current_directory + '\\traditional denoising.py'
    if len(viewer.layers.selection) == 0:
        window = QWidget()
        cp = QDesktopWidget().availableGeometry().center()
        window.setGeometry(cp.x() - 200, cp.y() - 100, 400, 200)
        QMessageBox.warning(window, 'Warning', 'Please select a layer or import an image first')
    else:
        try:
            subprocess.run(['python', script_path1, '--test'])
            subprocess.run(['python', script_path2, '--adaptive'])
        except Exception as e:
            print(f'执行脚本时发生错误：{e}')
        image_data = io.imread(current_directory + '\\val\\traditional improvement/Non_local_means_improvement.tif')
        viewer.add_image(image_data, scale=(5, 1, 1), colormap='red')
        delete_files_in_folder(current_directory + "\\val\\temp_png")
        delete_files_in_folder(current_directory + '\\val\\traditional improvement')
adaptive_denoising.clicked.connect(run_adaptive_denoising)

#小波变换降噪算法
wavelet_denoising = QPushButton('Wavelet denoising')
def run_wavelet_denoising():
    selected_layers = viewer.layers.selection
    for layer in selected_layers:
        napari.save_layers(current_directory + '\\val\\temp.tif', [layer])
    script_path1 = current_directory + '\\TiffToPng.py'
    script_path2 = current_directory + '\\traditional denoising.py'
    if len(viewer.layers.selection) == 0:
        window = QWidget()
        cp = QDesktopWidget().availableGeometry().center()
        window.setGeometry(cp.x() - 200, cp.y() - 100, 400, 200)
        QMessageBox.warning(window, 'Warning', 'Please select a layer or import an image first')
    else:
        try:
            subprocess.run(['python', script_path1, '--test'])
            subprocess.run(['python', script_path2, '--wavelet'])
        except Exception as e:
            print(f'执行脚本时发生错误：{e}')
        image_data = io.imread(current_directory + '\\val\\traditional improvement\\wavelet_improvement.tif')
        viewer.add_image(image_data, scale=(5, 1, 1), colormap='red')
        delete_files_in_folder(current_directory + "\\val\\temp_png")
        delete_files_in_folder(current_directory + '\\val\\traditional improvement')
wavelet_denoising.clicked.connect(run_wavelet_denoising)

#直方图均衡化算法
histogram_equalization = QPushButton('histogram equalization')
def run_histogram_equalization():
    selected_layers = viewer.layers.selection
    for layer in selected_layers:
        napari.save_layers(current_directory + '\\val\\temp.tif', [layer])
    script_path1 = current_directory + '\\TiffToPng.py'
    script_path2 = current_directory + '\\Histogram_equalisation.py'
    if len(viewer.layers.selection) == 0:
        window = QWidget()
        cp = QDesktopWidget().availableGeometry().center()
        window.setGeometry(cp.x() - 200, cp.y() - 100, 400, 200)
        QMessageBox.warning(window, 'Warning', 'Please select a layer or import an image first')
    else:
        try:
            subprocess.run(['python', script_path1, '--test'])
            subprocess.run(['python', script_path2])
        except Exception as e:
            print(f'执行脚本时发生错误：{e}')
        image_data = io.imread(current_directory + "\\val\\traditional improvement\\histogram_equalization.tif")
        viewer.add_image(image_data, scale=(5, 1, 1), colormap='red')
        delete_files_in_folder(current_directory + "\\val\\temp_png")
        delete_files_in_folder(current_directory + '\\val\\traditional improvement')
histogram_equalization.clicked.connect(run_histogram_equalization)

Remove_outliers = QPushButton('Remove outliers')
def run_Remove_outliers():
    selected_layers = viewer.layers.selection
    for layer in selected_layers:
        napari.save_layers(current_directory + '\\val\\temp.tif', [layer])
    script_path1 = current_directory + '\\TiffToPng.py'
    script_path2 = current_directory + '\\Remove_outliers.py'
    if len(viewer.layers.selection) == 0:
        window = QWidget()
        cp = QDesktopWidget().availableGeometry().center()
        window.setGeometry(cp.x() - 200, cp.y() - 100, 400, 200)
        QMessageBox.warning(window, 'Warning', 'Please select a layer or import an image first')
    else:
        try:
            subprocess.run(['python', script_path1, '--test'])
            subprocess.run(['python', script_path2])
        except Exception as e:
            print(f'执行脚本时发生错误：{e}')
        image_data = io.imread(current_directory + "\\val\\traditional improvement\\remove_outliers.tif")
        viewer.add_image(image_data, scale=(5, 1, 1), colormap='red')
        delete_files_in_folder(current_directory + "\\val\\temp_png")
        delete_files_in_folder(current_directory + '\\val\\traditional improvement')
Remove_outliers.clicked.connect(run_Remove_outliers)


#Despeckle
Despeckle = QPushButton('Despeckle')
def run_Despeckle():
    selected_layers = viewer.layers.selection
    for layer in selected_layers:
        napari.save_layers(current_directory + '\\val\\temp.tif', [layer])
    script_path1 = current_directory + '\\TiffToPng.py'
    script_path2 = current_directory + '\\Despeckle.py'
    if len(viewer.layers.selection) == 0:
        window = QWidget()
        cp = QDesktopWidget().availableGeometry().center()
        window.setGeometry(cp.x() - 200, cp.y() - 100, 400, 200)
        QMessageBox.warning(window, 'Warning', 'Please select a layer or import an image first')
    else:
        try:
            subprocess.run(['python', script_path1, '--test'])
            subprocess.run(['python', script_path2])
        except Exception as e:
            print(f'执行脚本时发生错误：{e}')
        image_data = io.imread(current_directory + "\\val\\traditional improvement\\Despeckled.tif")
        viewer.add_image(image_data, scale=(5, 1, 1), colormap='red')
        delete_files_in_folder(current_directory + "\\val\\temp_png")
        delete_files_in_folder(current_directory + '\\val\\traditional improvement')
Despeckle.clicked.connect(run_Despeckle)



layout = QFormLayout()
layout.addRow(file_label,file_open)
layout.addRow(Save_label,SaveTif)
layout.addRow(Save_snapshot_label,Save_snapshot)
layout.addWidget(AI_segmentation)
layout.addWidget(denoising)
layout.addWidget(adaptive_denoising)
layout.addWidget(wavelet_denoising)
layout.addWidget(histogram_equalization)
layout.addWidget(Despeckle)
layout.addWidget(Remove_outliers)
layout.addRow(z_height_label, z_height_Edit)
horizontal_layout = QHBoxLayout()
horizontal_layout.addWidget(background_label)
horizontal_layout.addWidget(black_background)
horizontal_layout.addWidget(white_background)
layout.addRow(horizontal_layout)
widget.setLayout(layout)

# 添加自定义窗口到napari查看器
viewer.window.add_dock_widget(widget, area='right', name='Custom Controls')

# 显示napari查看器
napari.run()
