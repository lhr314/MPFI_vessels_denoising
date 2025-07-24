# 作   者：罗 H R
# 开发时间：2024/2/2 17:41
import networks
import argparse
import utils
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from datasets import PairedImage_generate
import numpy as np
import cv2
import time

def get_opt():
    parser = argparse.ArgumentParser()
    # Parameters for testing
    parser.add_argument('--batch_size', type=int, default=8, help='batch size of testing')
    parser.add_argument('--cuda', action='store_true', help='use GPU computation')
    parser.add_argument('--rootdir', type=str, default='val/', help='root directory of the dataset')
    parser.add_argument('--n_cpu', type=int, default=4, help='number of cpu threads to use during batch generation')
    parser.add_argument('--U_net', action='store_true', help='use U-net')
    parser.add_argument('--ResU_net', action='store_true', help='use ResU-net')
    parser.add_argument('--NestedU_net', action='store_true', help='use Nested U-net')
    parser.add_argument('--FCN_Resnet', action='store_true', help='use FCN-Resnet')
    parser.add_argument('--ResU_net_Transformer', action='store_true', help='use ResU_net_Transformer')
    parser.add_argument('--Segmenter', action='store_true', help='use Segmenter')
    parser.add_argument('--MaskRCNN', action='store_true', help='use MaskRCNN')
    parser.add_argument('--DeeplabV3', action='store_true', help='use DeeplabV3')
    # Model parameters
    parser.add_argument('--sizeh', type=int, default=512, help='size of the image')
    parser.add_argument('--sizew', type=int, default=512, help='size of the image')
    parser.add_argument('--input_nc', type=int, default=1, help='number of input channels')
    parser.add_argument('--output_nc', type=int, default=1, help='number of output channels')
    parser.add_argument('--ngf', type=int, default=64, help='number of filters in the generator')
    parser.add_argument('--ndf', type=int, default=64, help='number of filters in the discriminator')
    parser.add_argument('--dropout', type=bool, default=False, help='whether to use dropout')
    parser.add_argument('--n_res', type=int, default=9, help='number of resNet blocks')
    parser.add_argument('--net_segmentation', type=str, default='model_metrics/net_segmentation.pth', help='path of the parameters of the generator A')

    opt = parser.parse_args()
    return opt

def main():
    opt = get_opt()
    device = torch.device("cuda") if opt.cuda else torch.device("cpu")

    # 定义具体的训练网络
    # 定义用于血管分割的神经网络
    if opt.U_net:
        net = networks.U_net(opt.input_nc, opt.output_nc)
    elif opt.NestedU_net:
        net = networks.NestedUNet(opt.input_nc, opt.output_nc)
    elif opt.ResU_net_Transformer:
        net = networks.Resnet34_Unet_transformer(opt.input_nc, opt.output_nc)
    elif opt.FCN_Resnet:
        net = networks.FCN_ResNet()
    elif opt.ResU_net:
        net = networks.Resnet34_Unet(opt.input_nc, opt.output_nc)
    elif opt.Segmenter:
        net = networks.Segmenter()
    elif opt.DeeplabV3:
        net = networks.DeeplabV3(opt.input_nc, opt.output_nc)
    net = net.to(device)
    if opt.cuda:
        net.cuda()

    #测试过程不需要梯度下降，故将优化器的梯度变化进行终止
    utils.set_requires_grad(net, False)
    net.eval()
    net.load_state_dict(torch.load(opt.net_segmentation))

    #加载数据
    transform = transforms.Compose([transforms.Resize((opt.sizeh, opt.sizew)),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5,), (0.5,))])
    dataloader = DataLoader(PairedImage_generate(opt.rootdir, transform=transform, mode='val'), batch_size=opt.batch_size,
                            shuffle=False, num_workers=opt.n_cpu)

    Tensor = torch.cuda.FloatTensor if opt.cuda else torch.Tensor
    # 记录开始时间
    start_time = time.time()
    # 测试
    for i, batch in enumerate(dataloader):
        name, image = batch
        image = image.to(device)
        #input = Tensor(len(image), opt.input_nc, opt.sizeh, opt.sizew)  # 自适应batch，防止样本数不是batch整倍数
        #ori = input.copy_(image)
        mask_pre = net(image)
        mask_pre = mask_pre > 0.5
        batch_size = len(name)
        for j in range(batch_size):
            image_name = name[j].split('\\')[-1]
            path = './val/temp_mask/' + image_name
            utils.save_image(mask_pre[j, :, :, :], path)
    # 记录结束时间
    end_time = time.time()
    # 计算生成时间
    inference_time = end_time - start_time
    print(f"生成时间: {inference_time} 秒")

if __name__ == '__main__':
    main()















