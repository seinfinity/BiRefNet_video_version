import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import fvcore.nn.weight_init as weight_init
from functools import partial
from einops import rearrange

from config import Config


config = Config()


class ResBlk(nn.Module):
    def __init__(self, channel_in=64, channel_out=64, channel_inter=64, dilation=config.dilation):
        super(ResBlk, self).__init__()
        channel_inter = channel_in // 4 if config.dec_channel_inter == 'adap' else 64
        self.conv_in = nn.Conv2d(channel_in, channel_inter, 3, 1, padding=dilation, dilation=dilation)
        self.relu_in = nn.ReLU(inplace=True)
        if config.dec_att:
            self.dec_att = AttentionModule(channel_in=channel_inter)
        self.conv_out = nn.Conv2d(channel_inter, channel_out, 3, 1, padding=dilation, dilation=dilation)
        if config.use_bn:
            self.bn_in = nn.BatchNorm2d(channel_inter)
            self.bn_out = nn.BatchNorm2d(channel_out)

    def forward(self, x):
        x = self.conv_in(x)
        if config.use_bn:
            x = self.bn_in(x)
        x = self.relu_in(x)
        if config.dec_att:
            x = self.dec_att(x)
        x = self.conv_out(x)
        if config.use_bn:
            x = self.bn_out(x)
        return x

class _ASPPModule(nn.Module):
    def __init__(self, channel_in, planes, kernel_size, padding, dilation):
        super(_ASPPModule, self).__init__()
        self.atrous_conv = nn.Conv2d(channel_in, planes, kernel_size=kernel_size,
                                            stride=1, padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU()

        # self._init_weight()

    def forward(self, x):
        x = self.atrous_conv(x)
        x = self.bn(x)

        return self.relu(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

class AttentionModule(nn.Module):
    def __init__(self, channel_in=64, output_stride=16):
        super(AttentionModule, self).__init__()
        self.down_scale = 4
        if output_stride == 16:
            dilations = [1, 6, 12, 18]
        elif output_stride == 8:
            dilations = [1, 12, 24, 36]
        else:
            raise NotImplementedError

        self.aspp1 = _ASPPModule(channel_in, 256 // self.down_scale, 1, padding=0, dilation=dilations[0])
        self.aspp2 = _ASPPModule(channel_in, 256 // self.down_scale, 3, padding=dilations[1], dilation=dilations[1])
        self.aspp3 = _ASPPModule(channel_in, 256 // self.down_scale, 3, padding=dilations[2], dilation=dilations[2])
        self.aspp4 = _ASPPModule(channel_in, 256 // self.down_scale, 3, padding=dilations[3], dilation=dilations[3])

        self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                             nn.Conv2d(channel_in, 256 // self.down_scale, 1, stride=1, bias=False),
                                             nn.BatchNorm2d(256 // self.down_scale),
                                             nn.ReLU())
        self.conv1 = nn.Conv2d(1280 // self.down_scale, 256 // self.down_scale, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(256 // self.down_scale)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self._init_weight()

    def forward(self, x):
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.global_avg_pool(x)
        x5 = F.interpolate(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat((x1, x2, x3, x4, x5), dim=1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        return self.dropout(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
