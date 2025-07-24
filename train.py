# 作   者：罗 H R
# 开发时间：2024/2/2 17:41
import networks
import argparse
import utils
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from datasets import PairedImage
import time
import numpy as np
from utils import DiceLoss,DiceBCELoss, FocalLoss,IoULoss
import torchmetrics
import csv

def get_opt():
    parser = argparse.ArgumentParser()
    # Model parameters
    parser.add_argument('--epoch', type=int, default=0, help='starting epoch')
    parser.add_argument('--n_epochs', type = int, default = 200, help = 'number of epochs with initial learning rate')
    parser.add_argument('--beta1', type = float, default = 0.5, help = 'momentum term of the Adam optimizer')
    parser.add_argument('--lr', type = float, default = 0.0004, help = 'initial learning rate')
    parser.add_argument('--batch_size', type = int, default = 8, help = 'batch size of training')
    parser.add_argument('--cuda', action='store_true', help='use GPU computation')
    parser.add_argument('--rootdir', type=str, default='datasets', help='root directory of the dataset')
    parser.add_argument('--n_cpu', type=int, default=4, help='number of cpu threads to use during batch generation')
    parser.add_argument('--U_net', action='store_true', help='use U-net')
    parser.add_argument('--ResU_net', action='store_true', help='use ResU-net')
    parser.add_argument('--NestedU_net', action='store_true', help='use Nested U-net')
    parser.add_argument('--FCN_Resnet', action='store_true', help='use FCN-Resnet')
    parser.add_argument('--ResU_net_Transformer', action='store_true', help='use ResU_net_Transformer')
    parser.add_argument('--Segmenter', action='store_true', help='use Segmenter')
    parser.add_argument('--DeeplabV3', action='store_true', help='use DeeplabV3')
    parser.add_argument('--pretrained', action='store_true', help='load pretrained weights')
    parser.add_argument('--α', type = int, default = 0.1, help = 'Control of positive and negative sample imbalance in Focal Loss')
    parser.add_argument('--gamma', type=int, default=2, help='Control of difficult and easy zone segmentation in Focal Loss')
    parser.add_argument('--Dice_Loss', action='store_true', help='use Dice Loss')
    parser.add_argument('--Focal_Loss', action='store_true', help='use Focal Loss')
    parser.add_argument('--Dice_Loss_Focal_Loss', action='store_true', help='use Dice Loss + Focal Loss')
    parser.add_argument('--α_Dice_loss',type = int, default = 1, help = 'Control of Dice loss as a percentage of total loss')
    parser.add_argument('--β_Focal_loss', type=int, default=0.1,help='Control of Focal loss as a percentage of total loss')
    parser.add_argument('--IoU_Loss', action='store_true', help='use IoU Loss')
    parser.add_argument('--cal_metrics', action='store_true', help='calculate metrics on the test set of every epoch')

    # image parameters
    parser.add_argument('--sizeh', type=int, default=512, help='size of the image')
    parser.add_argument('--sizew', type=int, default=512, help='size of the image')
    parser.add_argument('--input_nc', type = int, default = 1, help = 'number of input channels')
    parser.add_argument('--output_nc', type = int, default = 1, help = 'number of output channels')
    parser.add_argument('--dropout', type = bool, default = False, help = 'whether to use dropout')
    opt = parser.parse_args()
    return opt

def main():
    # 获取训练参数
    opt = get_opt()
    device = torch.device("cuda") if opt.cuda else torch.device("cpu")

    #定义具体的训练网络
    #定义用于血管分割的神经网络
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
    # 决定是否使用cuda
    if opt.cuda:
        net.cuda()
    #初始化网络权重
    utils.init_weight(net)

    #是否预训练
    if opt.pretrained:
        net.load_state_dict(torch.load('pretrained/net_segmentation.pth'))

    #定义损失函数，选用Diceloss\DiceBCEloss\Focalloss\Focalloss+Diceloss
    if opt.Dice_Loss:
        criterion = DiceLoss()
    elif opt.Focal_Loss:
        criterion = FocalLoss()
    elif opt.IoU_Loss:
        criterion = IoULoss()
    elif opt.Dice_Loss_Focal_Loss:
        criterion1 = FocalLoss()
        criterion2 = DiceLoss()

    # 定义优化器
    optimizer = torch.optim.Adam(net.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))  # 生成器网络的优化器

    # 创建学习率调度器，这里采用RLR自动调节调度器
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)

    #定义图像转换以及加载数据
    transform = transforms.Compose([transforms.Resize((opt.sizeh, opt.sizew)),  # 调整图像大小
                                transforms.ToTensor(),  # 将图像转换为张量
                                transforms.Normalize((0.5,), (0.5,))])  # 归一化图像数据
    dataloader = DataLoader(PairedImage(opt.rootdir, transform=transform, mode='train'), batch_size=opt.batch_size,
                            shuffle=True, num_workers=opt.n_cpu)  # 创建数据加载器

    if opt.cal_metrics:
        #加载测试集，用于每轮的指标测算
        dataloader_test = DataLoader(PairedImage(opt.rootdir, transform=transform, mode='val'), batch_size=opt.batch_size,
                           shuffle=True, num_workers=opt.n_cpu)

    # 用于存储每个epoch的损失的numpy数组
    loss_array = np.zeros(opt.n_epochs)  # 损失数组

    # 创建空列表存储过程中不同时间状态下的各指标
    G_loss_list = []
    G_Dice_list = []
    #训练过程
    for epoch in range(opt.epoch, opt.n_epochs):
        #当前轮次的指标列表
        loss_list = []
        start = time.strftime("%H:%M:%S")
        print("current epoch :", epoch, " start time :", start)
        loss_list.clear()
        for i, batch in enumerate(dataloader):
            if i % 30 == 1:
                print("current step: ", i)
                current = time.strftime("%H:%M:%S")
                print("current time :", current)
                print("last loss:", loss_list[-1])
            ori = batch['ori'].to(device)
            mask = batch['mask'].to(device)


            # 训练生成器
            utils.set_requires_grad([net], True)
            optimizer.zero_grad() #先将生成器梯度归零

            #生成预测图像
            mask_pre = net(ori)

            # 确保目标数据（标签）的值是二进制
            mask_binary = mask.clone()
            mask_binary[mask_binary > 0] = 1
            mask_binary[mask_binary <= 0] = 0

            #损失函数计算
            if opt.Dice_Loss_Focal_Loss:
                loss_Focal = criterion1(mask_pre, mask_binary)
                loss_Dice = criterion2(mask_pre, mask_binary)
                G_loss=opt.α_Dice_loss*loss_Dice+opt.β_Focal_loss*loss_Focal
                loss_list.append(G_loss.item())
                G_loss.backward()
                optimizer.step()
            else:
                G_loss = criterion(mask_pre, mask_binary)
                loss_list.append(G_loss.item())
                G_loss.backward()
                optimizer.step()
        G_loss_list.append(sum(loss_list) / len(loss_list))
        # 每批次结束后根据当前最后更新的loss_G更新学习率
        lr_scheduler.step(metrics=G_loss_list[-1])
        if opt.cal_metrics:
            Dice_list = []
            for j, batch_test in enumerate(dataloader_test):
                ori_test = batch_test['ori'].to(device)
                mask_test = batch_test['mask'].to(device)
                mask_test_pre = net(ori_test)
                mask_test_pre[mask_test_pre > 0.5] = 1
                mask_test_pre[mask_test_pre <= 0.5] = 0
                mask_test_pre_binary = mask_test_pre.clone()
                #mask_test_pre_binary[mask_test_pre_binary > 0] = 1
                #mask_test_pre_binary[mask_test_pre_binary <= 0] = 0

                mask_test_binary = mask_test.clone()
                mask_test_binary[mask_test_binary > 0] = 1
                mask_test_binary[mask_test_binary <= 0] = 0
                # 计算当前的测试集指标
                Dice_list.append((torchmetrics.functional.dice(preds=mask_test_pre_binary, target=mask_test_binary.to(torch.int32))).item())
            G_Dice_list.append(sum(Dice_list)/len(Dice_list))
            print(f'current dice = {G_Dice_list[-1]}')
            del Dice_list
        if opt.cal_metrics:
            if epoch ==0:
                max_Dice=G_Dice_list[-1]
            else :
                if G_Dice_list[-1] > max_Dice:
                    max_Dice=G_Dice_list[-1]
                    #若Dice值最优存储当前时间点的模型参数
                    torch.save(net.state_dict(), 'model_metrics/net_segmentation.pth')
            print(f'optimal dice = {max_Dice}')
        else:
            if epoch ==0:
                min_loss=G_loss_list[-1]
            else :
                if G_loss_list[-1] < min_loss:
                    min_loss=G_loss_list[-1]
                    #若DSC值最优存储当前时间点的模型参数
                    torch.save(net.state_dict(), 'model_metrics/net_segmentation.pth')
        #if epoch%10==0:
            #torch.save(net.state_dict(), f'model_metrics/net_segmentation{epoch}.pth')
        # checkpoint = {'epoch': epoch,
        #              'optimizer': optimizer.state_dict(),
        #              'lr_scheduler': lr_scheduler.state_dict()}
        # torch.save(checkpoint, 'model_metrics/checkpoint.pth')
        #更新并记录损失函数
        loss_array[epoch] = sum(loss_list) / len(loss_list)
        np.savetxt('model_metrics/loss.txt', loss_array)
        end = time.strftime("%H:%M:%S")
        print("current epoch :", epoch, " end time :", end)
        print("current loss :", loss_array[epoch])
        del loss_list
    def write_metric(csv_file_path,list,title):
        with open(csv_file_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([title])  # 写入标题行
            for score in list:
                writer.writerow([score])  # 写入F1分数
    if opt.cal_metrics:
        # 将指标写入CSV文件
        DSC_csv_file_path = "model_metrics/Dice.csv"
        write_metric(DSC_csv_file_path, G_Dice_list, "Dice")
    Loss_csv_file_path = "model_metrics/loss.csv"
    write_metric(Loss_csv_file_path, G_loss_list, "Loss")

if __name__ == "__main__":
    main()