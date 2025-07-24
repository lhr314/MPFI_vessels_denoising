# 作   者：罗 H R
# 开发时间：2024/2/2 17:41
from torch.utils.data import Dataset
from PIL import Image
import glob
import os

#加载降噪数据集
class ImageDataset(Dataset):
    # 初始化函数，用于设置数据集的参数和加载文件路径
    def __init__(self, root, transform=None, mode='train'):
        self.transform = transform  # 图像转换函数
        self.mode = mode  # 模式，train表示训练集
        if self.mode == 'test':
            self.files_A = sorted(glob.glob(os.path.join('val','temp_separation', 'temp_noise','*.*')))  # 获取A文件夹下的所有文件路径并排序
        if self.mode == 'train':
            self.files_A = sorted(glob.glob(os.path.join(root, 'Noise Separation', 'temp_noise', '*.*')))  # 获取A文件夹下的所有文件路径并排序
            self.files_B = sorted(glob.glob(os.path.join(root, 'Noise Separation', 'noiseless', '*.*')))  # 如果是训练模式，获取B文件夹下的所有文件路径并排序

    # 获取数据集中的一项
    def __getitem__(self, index):
        # 从A文件夹中读取图像并进行转换
        item_A = self.transform(Image.open(self.files_A[index]))
        if self.mode == 'train':
            item_B = self.transform(Image.open(self.files_B[index]))
            return {'A': item_A, 'B': item_B}
        else:
            return self.files_A[index], item_A

    # 获取数据集的长度
    def __len__(self):
            return len(self.files_A)

#加载分割
class PairedImage(Dataset):
    def __init__(self, root, transform = None, mode = 'train'):
        self.transform = transform
        self.mode = mode
        if self.mode == 'train':
            self.files_B = sorted(glob.glob(os.path.join(root, 'train', 'mask', '*.*')))
            self.files_A = sorted(glob.glob(os.path.join(root, 'train', 'ori', '*.*')))
        else:
            self.files_B = sorted(glob.glob(os.path.join("val","Y_mask_test", '*.*')))
            self.files_A = sorted(glob.glob(os.path.join("val","temp_png", '*.*')))


    def __getitem__(self, index):
        item_A = self.transform(Image.open(self.files_A[index]))
        if self.mode == 'train':
            item_B = self.transform(Image.open(self.files_B[index]))
            return {'ori': item_A, 'mask': item_B}
        else:
            item_B = self.transform(Image.open(self.files_B[index]))
            return {'ori': item_A, 'mask': item_B}

    def __len__(self):
        return len(self.files_A)

class PairedImage_generate(Dataset):
    def __init__(self, root, transform = None, mode = 'train'):
        self.transform = transform
        self.mode = mode
        if self.mode == 'train':
            self.files_B = sorted(glob.glob(os.path.join(root, 'train', 'mask', '*.*')))
            self.files_A = sorted(glob.glob(os.path.join(root, 'train', 'ori', '*.*')))
        else:
            self.files_A = sorted(glob.glob(os.path.join("val","temp_png", '*.*')))

    def __getitem__(self, index):
        item_A = self.transform(Image.open(self.files_A[index]))
        if self.mode == 'train':
            item_B = self.transform(Image.open(self.files_B[index]))
            return {'ori': item_A, 'mask': item_B}
        else:
            return self.files_A[index], item_A

    def __len__(self):
        return len(self.files_A)

class PairedImage_metrics(Dataset):
    def __init__(self, root, transform = None):
        self.transform = transform
        self.files_B = sorted(glob.glob(os.path.join("val","temp_mask", '*.*')))
        self.files_A = sorted(glob.glob(os.path.join("val","Y_mask_test", '*.*')))
    def __getitem__(self, index):
        item_A = self.transform(Image.open(self.files_A[index]))
        item_B = self.transform(Image.open(self.files_B[index]))
        return {'mask_ori': item_A, 'mask_pre': item_B}
    def __len__(self):
        return len(self.files_A)