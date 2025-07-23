# 作   者：罗 H R
# 开发时间：2024/2/2 17:42
import torch
from torch.nn import init
import torch.nn as nn
import random
import torchvision.transforms as transforms
import torch.nn.functional as F
import numpy as np
from PIL import Image

#初始化网络权重
def init_weight(net, init_gain = 0.02):
    def init_func(m):
        #获取当前层的类名
        classname = m.__class__.__name__
        #如果当前层具有 'weight' 属性，并且是卷积层(Conv)或全连接层(Linear)
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            # 对权重进行正态分布初始化，均值为0，标准差为init_gain
            init.normal_(m.weight.data, 0.0, init_gain)
            # 如果当前层具有 'bias' 属性且偏置项不为None，则对偏置项进行常数初始化，值为0
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        # 如果当前层是 BatchNormalization2d 层
        elif classname.find('BatchNorm2d') != -1:
            # 对权重进行正态分布初始化，均值为1，标准差为init_gain
            init.normal_(m.weight.data, 1.0, init_gain)
            # 对偏置项进行常数初始化，值为0
            init.constant_(m.bias.data, 0.0)
    net.apply(init_func)

#定义diceloss损失函数
class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):

        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = torch.sigmoid(inputs)

        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)

        return 1 - dice

class DiceBCELoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceBCELoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):

        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = torch.sigmoid(inputs)

        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice_loss = 1 - (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        Dice_BCE = BCE + dice_loss

        return Dice_BCE

class DiceBCELoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceBCELoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):

        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = torch.sigmoid(inputs)

        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice_loss = 1 - (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        Dice_BCE = BCE + dice_loss

        return Dice_BCE


class IoULoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(IoULoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        # comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)

        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        # intersection is equivalent to True Positive count
        # union is the mutually inclusive area of all labels & predictions
        intersection = (inputs * targets).sum()
        total = (inputs + targets).sum()
        union = total - intersection

        IoU = (intersection + smooth) / (union + smooth)

        return 1 - IoU

# 梯度下降过程函数
def set_requires_grad(nets, requires_grad = False):
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad

# 生成对抗网络（GAN）的损失函数，主要用于训练生成器和判别器的模型
class GANLoss(nn.Module):
    def __init__(self, target_real_label = 1.0, target_fake_label = 0.0):
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.loss = nn.MSELoss()

    def get_target_tensor(self, prediction, target_is_real):
        # Return a tensor filled with ground-truth label, and has the same size as the prediction
        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label

        return target_tensor.expand_as(prediction)

    def __call__(self, prediction, target_is_real):
        target_tensor = self.get_target_tensor(prediction, target_is_real)
        return self.loss(prediction, target_tensor)

#图像池类
class ImagePool():
    def __init__(self, pool_size=50):
        self.pool_size = pool_size
        if self.pool_size > 0:
            # Create an empty pool
            self.num_imgs = 0
            self.images = []
    def query(self, images):
        # return an image from the image pool
        # If the pool size is 0, just return the input images
        if self.pool_size == 0:
            return images
        return_images = []
        for image in images:
            image = torch.unsqueeze(image.data, 0)
            if self.num_imgs < self.pool_size:
                # If the pool is not full, insert the current image
                self.num_imgs += 1
                self.images.append(image)
                return_images.append(image)
            else:
                p = random.uniform(0, 1)
                if p > 0.5:
                    # return a random image, and insert current image in the pool
                    random_id = random.randint(0, self.pool_size - 1)
                    tmp = self.images[random_id].clone()
                    self.images[random_id] = image
                    return_images.append(tmp)
                else:
                    # return current image
                    return_images.append(image)
        return_images = torch.cat(return_images, 0)
        return return_images

# 将图像从张量转化回png图像
def save_image(tensor, name):
    unloader = transforms.ToPILImage()
    image = tensor.cpu().clone()
    image = image * 0.5 + 0.5 # 将张量的值从[-1,1]映射到[0,1]范围
    image = image.squeeze(0)  # 去除张量中的虚拟批次维度，如果存在的话
    image = unloader(image) # 使用 ToPILImage 转换器将张量转换为 PIL 图像
    np_img = np.array(image)
    np_img[tensor[:][0].cpu().numpy() == 0] = 0
    image=Image.fromarray(np_img)
    image.save(name, "PNG")# 保存图像为PNG格式


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.05, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        if self.reduction == 'mean':
            return torch.mean(focal_loss)
        elif self.reduction == 'sum':
            return torch.sum(focal_loss)
        else:
            return focal_loss