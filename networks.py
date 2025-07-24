# 作   者：罗 H R
# 开发时间：2024/2/2 17:41
import torch.nn as nn
import torch
from torchvision import models
from torch.nn import TransformerEncoder, TransformerEncoderLayer, TransformerDecoder,TransformerDecoderLayer
from timm.models.layers import DropPath, to_2tuple
import math
import torch.nn.functional as F
from einops import rearrange
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torchvision


#1. U-net：
#  定义双卷积层
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ]
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

#  定义降采样
class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()
        layers = [
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        ]
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

# The building block of U-Net
# Up-sampling layers of U-Net

#  定义上采样
class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Up, self).__init__()
        self.conv1 = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv2 = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.conv1(x1)
        x = torch.cat((x2, x1), dim=1)
        x = self.conv2(x)
        return x

class U_net(nn.Module):
    def __init__(self, in_channels, out_channels, out_features = 16):
        super(U_net, self).__init__()
        self.conv1 = DoubleConv(in_channels, out_features)
        self.conv2 = Down(out_features, out_features * 2)
        self.conv3 = Down(out_features * 2, out_features * 4)
        self.conv4 = Down(out_features * 4, out_features * 8)
        self.conv5 = Down(out_features * 8, out_features * 16)
        self.deconv4 = Up(out_features * 16, out_features * 8)
        self.deconv3 = Up(out_features * 8, out_features * 4)
        self.deconv2 = Up(out_features * 4, out_features * 2)
        self.deconv1 = Up(out_features * 2, out_features)
        layers=  [
            nn.Conv2d(in_channels=out_features, out_channels=out_channels, kernel_size=3, padding=1),
            nn.Tanh()
        ]
        self.output = nn.Sequential(*layers)


    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        x5 = self.conv5(x4)
        out = self.deconv4(x5, x4)
        out = self.deconv3(out, x3)
        out = self.deconv2(out, x2)
        out = self.deconv1(out, x1)
        out = self.output(out)
        return out

#2. NestedU-net
# VGGBlock定义
class conv_block_nested(nn.Module):

    def __init__(self, in_ch, mid_ch, out_ch):
        super(conv_block_nested, self).__init__()
        self.activation = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_ch, mid_ch, kernel_size=3, padding=1, bias=True)
        self.bn1 = nn.BatchNorm2d(mid_ch)
        self.conv2 = nn.Conv2d(mid_ch, out_ch, kernel_size=3, padding=1, bias=True)
        self.bn2 = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation(x)

        x = self.conv2(x)
        x = self.bn2(x)
        output = self.activation(x)
        return output

class NestedUNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(NestedUNet,self).__init__()

        n1 = 32
        nb_filter = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]

        self.pool = nn.MaxPool2d(2, 2)
        self.Up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv0_0 = conv_block_nested(in_channels, nb_filter[0], nb_filter[0])
        self.conv1_0 = conv_block_nested(nb_filter[0], nb_filter[1], nb_filter[1])
        self.conv2_0 = conv_block_nested(nb_filter[1], nb_filter[2], nb_filter[2])
        self.conv3_0 = conv_block_nested(nb_filter[2], nb_filter[3], nb_filter[3])
        self.conv4_0 = conv_block_nested(nb_filter[3], nb_filter[4], nb_filter[4])

        self.conv0_1 = conv_block_nested(nb_filter[0] + nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_1 = conv_block_nested(nb_filter[1] + nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_1 = conv_block_nested(nb_filter[2] + nb_filter[3], nb_filter[2], nb_filter[2])
        self.conv3_1 = conv_block_nested(nb_filter[3] + nb_filter[4], nb_filter[3], nb_filter[3])

        self.conv0_2 = conv_block_nested(nb_filter[0] * 2 + nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_2 = conv_block_nested(nb_filter[1] * 2 + nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_2 = conv_block_nested(nb_filter[2] * 2 + nb_filter[3], nb_filter[2], nb_filter[2])

        self.conv0_3 = conv_block_nested(nb_filter[0] * 3 + nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_3 = conv_block_nested(nb_filter[1] * 3 + nb_filter[2], nb_filter[1], nb_filter[1])

        self.conv0_4 = conv_block_nested(nb_filter[0] * 4 + nb_filter[1], nb_filter[0], nb_filter[0])

        self.final = nn.Conv2d(nb_filter[0], out_channels, kernel_size=1)



    def forward(self, input):
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x0_1 = self.conv0_1(torch.cat([x0_0, self.Up(x1_0)], 1))

        x2_0 = self.conv2_0(self.pool(x1_0))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.Up(x2_0)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.Up(x1_1)], 1))

        x3_0 = self.conv3_0(self.pool(x2_0))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.Up(x3_0)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.Up(x2_1)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.Up(x1_2)], 1))

        x4_0 = self.conv4_0(self.pool(x3_0))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.Up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.Up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.Up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.Up(x1_3)], 1))

        output = self.final(x0_4)
        return output

#3. NestedUnet_LSTM

class NestedUNet_LSTM(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        nb_filter = [32, 64, 128, 256, 512]

        self.pool = nn.MaxPool2d(2, 2)
        self.Up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv0_0 = conv_block_nested(in_channels, nb_filter[0], nb_filter[0])
        self.conv1_0 = conv_block_nested(nb_filter[0], nb_filter[1], nb_filter[1])
        self.conv2_0 = conv_block_nested(nb_filter[1], nb_filter[2], nb_filter[2])
        self.conv3_0 = conv_block_nested(nb_filter[2], nb_filter[3], nb_filter[3])
        self.conv4_0 = conv_block_nested(nb_filter[3], nb_filter[4], nb_filter[4])

        self.conv0_1 = conv_block_nested(nb_filter[0] + nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_1 = conv_block_nested(nb_filter[1] + nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_1 = conv_block_nested(nb_filter[2] + nb_filter[3], nb_filter[2], nb_filter[2])
        self.conv3_1 = conv_block_nested(nb_filter[3] + nb_filter[4], nb_filter[3], nb_filter[3])

        self.conv0_2 = conv_block_nested(nb_filter[0] * 2 + nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_2 = conv_block_nested(nb_filter[1] * 2 + nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_2 = conv_block_nested(nb_filter[2] * 2 + nb_filter[3], nb_filter[2], nb_filter[2])

        self.conv0_3 = conv_block_nested(nb_filter[0] * 3 + nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_3 = conv_block_nested(nb_filter[1] * 3 + nb_filter[2], nb_filter[1], nb_filter[1])

        self.conv0_4 = conv_block_nested(nb_filter[0] * 4 + nb_filter[1], nb_filter[0], nb_filter[0])

        self.final = nn.Conv2d(nb_filter[0], out_channels, kernel_size=1)
        # LSTM层
        self.lstm = nn.LSTM(input_size=nb_filter[4], hidden_size=nb_filter[4], num_layers=5, batch_first=True)
        # 添加分割标志的卷积层
        self.separator_conv = nn.Conv2d(in_channels=nb_filter[4], out_channels=1, kernel_size=3, padding=1)
        self.separator_activation = nn.Sigmoid()


    def forward(self, input):
        x0_0 = self.conv0_0(input)
        lstm_input = x0_0.permute(0, 2, 3, 1).contiguous().view(x0_0.size(0), -1, x0_0.size(1))  # 调整形状以适应LSTM
        lstm_output, _ = self.lstm(lstm_input)
        lstm_output = lstm_output.view(x0_0.size(0), x0_0.size(2), x0_0.size(3), x0_0.size(1)).permute(0, 3, 1, 2)
        # 添加分割标志
        separator = self.separator_activation(self.separator_conv(x0_0))
        separator = separator.expand_as(x0_0)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x0_1 = self.conv0_1(torch.cat([x0_0, self.Up(x1_0)], 1))

        x2_0 = self.conv2_0(self.pool(x1_0))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.Up(x2_0)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.Up(x1_1)], 1))

        x3_0 = self.conv3_0(self.pool(x2_0))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.Up(x3_0)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.Up(x2_1)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.Up(x1_2)], 1))

        x4_0 = self.conv4_0(self.pool(x3_0))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.Up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.Up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.Up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.Up(x1_3)], 1))

        output = self.final(x0_4)
        return output

#4. ResU-net
class expansive_block(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels):
        super(expansive_block, self).__init__()

        self.block = nn.Sequential(
            nn.Conv2d(kernel_size=(3, 3), in_channels=in_channels, out_channels=mid_channels, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(mid_channels),
            nn.Conv2d(kernel_size=(3, 3), in_channels=mid_channels, out_channels=out_channels, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, d, e=None):
        d = F.interpolate(d, scale_factor=2, mode='bilinear', align_corners=True)
        # concat

        if e is not None:
            cat = torch.cat([e, d], dim=1)
            out = self.block(cat)
        else:
            out = self.block(d)
        return out


def final_block(in_channels, out_channels):
    block = nn.Sequential(
        nn.Conv2d(kernel_size=(3, 3), in_channels=in_channels, out_channels=out_channels, padding=1),
        nn.ReLU(),
        nn.BatchNorm2d(out_channels),
    )
    return block


class Resnet34_Unet(nn.Module):

    def __init__(self, in_channel, out_channel, pretrained=False):
        super(Resnet34_Unet, self).__init__()

        self.resnet = models.resnet34(pretrained=pretrained)
        self.layer0 = nn.Sequential(
            self.resnet.conv1,
            self.resnet.bn1,
            self.resnet.relu,
            self.resnet.maxpool
        )

        # Encode
        self.layer1 = self.resnet.layer1
        self.layer2 = self.resnet.layer2
        self.layer3 = self.resnet.layer3
        self.layer4 = self.resnet.layer4

        # Bottleneck
        self.bottleneck = torch.nn.Sequential(
            nn.Conv2d(kernel_size=(3, 3), in_channels=512, out_channels=1024, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(1024),
            nn.Conv2d(kernel_size=(3, 3), in_channels=1024, out_channels=1024, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(1024),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        )

        # Decode
        self.conv_decode4 = expansive_block(1024 + 512, 512, 512)
        self.conv_decode3 = expansive_block(512 + 256, 256, 256)
        self.conv_decode2 = expansive_block(256 + 128, 128, 128)
        self.conv_decode1 = expansive_block(128 + 64, 64, 64)
        self.conv_decode0 = expansive_block(64, 32, 32)
        self.conv_decode = expansive_block(32, 16, 16)
        self.final_layer = final_block(16, out_channel)

    def forward(self, x):
        if x.size(1) == 1:
            x=x.repeat(1, 3, 1, 1)
        x = self.layer0(x)
        # Encode
        encode_block1 = self.layer1(x)
        encode_block2 = self.layer2(encode_block1)
        encode_block3 = self.layer3(encode_block2)
        encode_block4 = self.layer4(encode_block3)

        # Bottleneck
        bottleneck = self.bottleneck(encode_block4)

        # Decode
        decode_block4 = self.conv_decode4(bottleneck, encode_block4)
        decode_block3 = self.conv_decode3(decode_block4, encode_block3)
        decode_block2 = self.conv_decode2(decode_block3, encode_block2)
        decode_block1 = self.conv_decode1(decode_block2, encode_block1)
        decode_block0 = self.conv_decode0(decode_block1)
        decode_block = self.conv_decode(decode_block0)
        final_layer = self.final_layer(decode_block)
        return final_layer

class Resnet34_Unet_transformer(nn.Module):

    def __init__(self, in_channel, out_channel, pretrained=False):
        super(Resnet34_Unet_transformer, self).__init__()

        self.resnet = models.resnet34(pretrained=pretrained)
        self.layer0 = nn.Sequential(
            self.resnet.conv1,
            self.resnet.bn1,
            self.resnet.relu,
            self.resnet.maxpool
        )

        # Encode
        self.layer1 = self.resnet.layer1
        self.layer2 = self.resnet.layer2
        self.layer3 = self.resnet.layer3
        self.layer4 = self.resnet.layer4

        # Bottleneck
        self.bottleneck = torch.nn.Sequential(
            nn.Conv2d(kernel_size=(3, 3), in_channels=512, out_channels=1024, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(1024),
            nn.Conv2d(kernel_size=(3, 3), in_channels=1024, out_channels=1024, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(1024),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        )
        # self-attention layers
        self.transformer_layer1 = nn.TransformerEncoderLayer(d_model=64, nhead=8, batch_first=True)
        self.transformer_layer2 = nn.TransformerEncoderLayer(d_model=1024, nhead=8, batch_first=True)


        # Transformer Encoder Layer1: batch sequence
        self.transformer_encoder1 = nn.TransformerEncoder(self.transformer_layer1, num_layers=6)

        # Transformer Encoder Layer2: pixels sequence
        self.transformer_encoder2 = nn.TransformerEncoder(self.transformer_layer2, num_layers=6)

        # Decode
        self.conv_decode4 = expansive_block(1024 + 512, 512, 512)
        self.conv_decode3 = expansive_block(512 + 256, 256, 256)
        self.conv_decode2 = expansive_block(256 + 128, 128, 128)
        self.conv_decode1 = expansive_block(128 + 64, 64, 64)
        self.conv_decode0 = expansive_block(64, 32, 32)
        self.conv_decode = expansive_block(32, 16, 16)
        self.final_layer = final_block(16, out_channel)
    def forward(self, x):
        if x.size(1) == 1:
            x=x.repeat(1, 3, 1, 1)
        x = self.layer0(x)
        # Encode
        encode_block1 = self.layer1(x)
        encode_block2 = self.layer2(encode_block1)
        encode_block3 = self.layer3(encode_block2)
        encode_block4 = self.layer4(encode_block3)

        # Bottleneck
        bottleneck = self.bottleneck(encode_block4)
        # Transformer Encoder1
        batch_size, num, height, width = bottleneck.size()
        bottleneck = bottleneck.view(batch_size, -1, height * width)
        bottleneck = bottleneck.permute(1, 0, 2)
        bottleneck = self.transformer_encoder1(bottleneck)
        bottleneck = bottleneck.permute(1, 0, 2)
        bottleneck = bottleneck.view(batch_size, -1, height, width)
        # Transformer Encoder2
        bottleneck = bottleneck.permute(0, 2, 3, 1)  # [batch, height, width, channels]
        bottleneck = bottleneck.view(bottleneck.size(0), -1, bottleneck.size(-1))  # [batch, seq_length, channels]
        bottleneck = self.transformer_encoder2(bottleneck)
        bottleneck = bottleneck.view(batch_size, num, height, width)  # reshape back to image size

        # Decode
        decode_block4 = self.conv_decode4(bottleneck, encode_block4)
        decode_block3 = self.conv_decode3(decode_block4, encode_block3)
        decode_block2 = self.conv_decode2(decode_block3, encode_block2)
        decode_block1 = self.conv_decode1(decode_block2, encode_block1)
        decode_block0 = self.conv_decode0(decode_block1)
        decode_block = self.conv_decode(decode_block0)
        final_layer = self.final_layer(decode_block)
        return final_layer

# 5. FCN-Resnet
class ResNet(nn.Module):

    def __init__(self, block, layers, out_stride, mult_grid):
        self.inplanes = 64
        super(ResNet, self).__init__()
        if out_stride == 8:
            stride = [2, 1, 1]
        elif out_stride == 16:
            stride = [2, 2, 1]
        elif out_stride == 32:
            stride = [2, 2, 2]
        # setting resnet last layer with dilation
        if mult_grid:
            if layers[-1] == 3:  # layers >= 50
                mult_grid = [2, 4, 6]
                mult_grid = [4, 8, 16]
            else:
                mult_grid = [2, 4]
                mult_grid = [4, 8]
        else:
            mult_grid = []

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=stride[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=stride[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=stride[2], dilation=mult_grid)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, dilation=[]):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        if dilation != []:
            layers.append(block(self.inplanes, planes, dilation[0], stride, downsample))
        else:
            layers.append(block(self.inplanes, planes, 1, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            if dilation != []:
                layers.append(block(self.inplanes, planes, dilation[i]))
            else:
                layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        blocks = []

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        blocks.append(x)
        x = self.layer2(x)
        blocks.append(x)
        x = self.layer3(x)
        blocks.append(x)
        x = self.layer4(x)
        blocks.append(x)

        return blocks
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, dilation=1, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride,
                     padding=dilation, dilation=dilation, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,
                     padding=dilation, dilation=dilation, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

def resnet34(out_stride=32, mult_grid=False):
    """Constructs a ResNet-34 model."""
    model = ResNet(BasicBlock, [3, 4, 6, 3], out_stride, mult_grid)

    return model
class FCN_ResNet(nn.Module):
    def __init__(self, num_classes=1):
        super(FCN_ResNet, self).__init__()

        self.backbone = resnet34(out_stride=32, mult_grid=False)

        self.conv_1 = nn.Conv2d(in_channels=512, out_channels=num_classes, kernel_size=1)
        self.conv_2 = nn.Conv2d(in_channels=256, out_channels=num_classes, kernel_size=1)
        self.conv_3 = nn.Conv2d(in_channels=128, out_channels=num_classes, kernel_size=1)
        self.conv_4 = nn.Conv2d(in_channels=64, out_channels=num_classes, kernel_size=1)

    def forward(self, x):
        if x.size(1) == 1:
            x=x.repeat(1, 3, 1, 1)
        layers = self.backbone(x)  # resnet 4 layers
        layers3 = self.conv_1(layers[3])
        layers3 = F.interpolate(layers3, layers[2].size()[2:], mode="bilinear", align_corners=True)
        layers2 = self.conv_2(layers[2])

        output = layers2 + layers3
        output = F.interpolate(output, layers[1].size()[2:], mode="bilinear", align_corners=True)
        layers1 = self.conv_3(layers[1])

        output = output + layers1
        output = F.interpolate(output, layers[0].size()[2:], mode="bilinear", align_corners=True)
        layers0 = self.conv_4(layers[0])

        output = output + layers0
        output = F.interpolate(output, x.size()[2:], mode="bilinear", align_corners=True)
        aux1 = F.interpolate(layers2, x.size()[2:], mode="bilinear", align_corners=True)
        return output

# 6. Segmenter

def padding(im, patch_size, fill_value=0):
    # make the image sizes divisible by patch_size
    H, W = im.size(2), im.size(3)
    pad_h, pad_w = 0, 0
    if H % patch_size > 0:
        pad_h = patch_size - (H % patch_size)
    if W % patch_size > 0:
        pad_w = patch_size - (W % patch_size)
    im_padded = im
    if pad_h > 0 or pad_w > 0:
        im_padded = F.pad(im, (0, pad_w, 0, pad_h), value=fill_value)
    return im_padded


def unpadding(y, target_size):
    H, W = target_size
    H_pad, W_pad = y.size(2), y.size(3)
    # crop predictions on extra pixels coming from padding
    extra_h = H_pad - H
    extra_w = W_pad - W
    if extra_h > 0:
        y = y[:, :, :-extra_h]
    if extra_w > 0:
        y = y[:, :, :, :-extra_w]
    return y


class PatchEmbed(nn.Module):  # 继承nn.Module
    """
    所有注释均采用VIT-base进行说明
    图像嵌入模块类
    2D Image to Patch Embedding
    """

    # 初始化函数，设置默认参数
    def __init__(self, img_size=512, patch_size=8, in_c=1, embed_dim=2048, norm_layer=None):
        super().__init__()  # 继承父类的初始化方法
        # 输入图像的size为512*512
        img_size = (img_size, img_size)
        # patch_size为8*8
        patch_size = (patch_size, patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        # 滑动窗口的大小为32*32， 512/8=64
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        # 图像切分的patch的总数为64*64=4096
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        # 使用一个卷积层来实现图像嵌入，输入维度为[BatchSize, 1, 512, 512]，输出维度为[BatchSize, 2048, 64, 64],
        # 计算公式 size= (512-64+2*0)/64 + 1= 8
        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=patch_size, stride=patch_size)
        # 如果norm_layer为True，则使用，否则忽略
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        # 获取BatchSize，Channels，Height，Width
        B, C, H, W = x.shape  # 输入批量的图像
        # VIT模型对图像的输入尺寸有严格要求，故需要检查是否为512*512，如果不是则报错提示
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        # flatten: [B, C, H, W] -> [B, C, HW]
        # transpose: [B, C, HW] -> [B, HW, C]
        # 将卷积嵌入后的图像特征图[BatchSize, 2048, 64,64],从第3维度开始展平，得到BatchSize*2048*4096；
        # 然后转置1，2两个维度,得到BatchSize*4096*2048
        x = self.proj(x).flatten(2).transpose(1, 2)
        # norm_layer层规范化
        x = self.norm(x)
        return x


class Attention(nn.Module):
    # 经过注意力层的特征的输入和输出维度相同
    def __init__(self,
                 dim,  # 输入token的维度
                 num_heads=8,  # multiHead中 head的个数
                 qkv_bias=False,  # 决定生成Q,K,V时是否使用偏置
                 qk_scale=None,
                 attn_drop_ratio=0.,
                 proj_drop_ratio=0.):
        super(Attention, self).__init__()
        # 设置多头注意力的注意力头的数目
        self.num_heads = num_heads
        # 针对每个head进行均分，它的Q,K,V对应的维度；
        head_dim = dim // num_heads
        # 放缩Q*（K的转置），就是根号head_dim(就是d(k))分之一，及和原论文保持一致
        self.scale = qk_scale or head_dim ** -0.5
        # 通过一个全连接层生成Q,K,V
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        # 定义dropout层
        self.attn_drop = nn.Dropout(attn_drop_ratio)
        # 通过一个全连接层实现将每一个头得到的注意力进行拼接
        self.proj = nn.Linear(dim, dim)
        # 使用dropout层
        self.proj_drop = nn.Dropout(proj_drop_ratio)

    def forward(self, x):
        # [batch_size, num_patches + 1, total_embed_dim]，其中，num_patches+1中的加1是为了保留class_token的位置
        B, N, C = x.shape  # [batch_size, 197, 768]

        # 生成Q,K,V；qkv(): -> [batch_size, num_patches + 1, 3 * total_embed_dim]
        # reshape: -> [batch_size, num_patches + 1, 3, num_heads, embed_dim_per_head],
        # 调换维度位置，permute: -> [3, batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # 将Q,K,V分离出来，切片后的Q,K,V的形状[batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        # 这个的维度相同，均为[BatchSize, 8, 197, 768]
        q, k, v = qkv[0], qkv[1], qkv[2]

        # transpose: -> [batch_size, num_heads, embed_dim_per_head, num_patches + 1]
        # @: multiply -> [batch_size, num_heads, num_patches + 1, num_patches + 1]
        # 即文章里的那个根据q,k, v 计算注意力的公式的Q*K的转置再除以根号dk
        attn = (q @ k.transpose(-2, -1)) * self.scale
        # 对得到的注意力结果的每一行进行softmax处理
        attn = attn.softmax(dim=-1)
        # 添加dropout层
        attn = self.attn_drop(attn)

        # @: multiply -> [batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        # transpose: -> [batch_size, num_patches + 1, num_heads, embed_dim_per_head]
        # reshape: -> [batch_size, num_patches + 1, total_embed_dim]
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)  # 乘V，通过reshape处理将每个head对应的结果进行拼接处理
        # 使用全连接层进行映射，维度不变，输入[BatchSize, 197, 768], 输出也相同
        x = self.proj(x)
        # 添加dropout层
        x = self.proj_drop(x)
        return x


class Mlp(nn.Module):
    """
    MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        # 注意当or两边存在None值时，输出是不为None那个
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        # MLP模块中的第一个全连接层，输入维度in_features, 输出维度hidden_features
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        # MLP模块中的第二个全连接层，输入维度hidden_features, 输出维度out_features
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Segmenter(nn.Module):
    def __init__(
        self,
        dim =2048,
        patch_size=8,
        n_cls=1,
        in_channels=1,
        out_channels=1
    ):
        super().__init__()
        self.dim = dim
        self.n_cls = n_cls
        self.patch_size = patch_size
        self.PatchEmbed = PatchEmbed()
        self.Attention = Attention(dim=self.dim, num_heads=8)
        self.Mlp = Mlp(in_features=self.dim, hidden_features=self.dim*4, act_layer=nn.GELU, drop=0.0)
        self.head = nn.Linear(2048, self.n_cls)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.dim))
        self.num_tokens = 1
        self.pos_embed = nn.Parameter(torch.zeros(1, 4096 + self.num_tokens, self.dim))
        self.norm = nn.LayerNorm(self.dim)
    def forward(self, im):
        batch_size = im.size(0)
        H_ori, W_ori = im.size(2), im.size(3)
        im = padding(im, self.patch_size)
        H, W = im.size(2), im.size(3)
        im = self.PatchEmbed(im)
        cls_token = self.cls_token.expand(im.shape[0], -1, -1)
        im = torch.cat((cls_token, im), dim=1)
        im = im + self.pos_embed
        x = im + self.Attention(self.norm(im))
        x = x + self.Mlp(self.norm(x))
        x = self.head(x)
        GS = H_ori // self.patch_size
        num_extra_tokens = 1
        x = x[:, num_extra_tokens:]
        masks = rearrange(x, "b (h w) c -> b c h w", h=GS)
        masks = F.interpolate(masks, size=(H, W), mode="bilinear")
        masks = unpadding(masks, (H_ori, W_ori))
        return masks

class DeeplabV3(nn.Module):
    def __init__(self,int_channels=1, out_channels=1):
        super().__init__()
        self.model=models.segmentation.deeplabv3_resnet50(num_classes=1)
    def forward(self, x):
        if x.size(1) == 1:
            x=x.repeat(1, 3, 1, 1)
        masks = self.model(x)
        return masks['out']
