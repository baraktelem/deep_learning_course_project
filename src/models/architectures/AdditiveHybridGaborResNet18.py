import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.layers.BasicBlock import BasicBlock
from src.models.layers.AdditiveHybridGaborLayer import AdditiveHybridGaborLayer

class AdditiveHybridGaborResNet(nn.Module):
    def __init__(self, blocks, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.conv1 = AdditiveHybridGaborLayer(3, 64, conv_kernel_size=3, gabor_kernel_size=7, ratio=1, stride=1)
        self.bn_std = nn.BatchNorm2d(64)
        self.bn_param = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(blocks[0], 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(blocks[1], 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(blocks[2], 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(blocks[3], 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layer = []
        for s in strides:
            layer.append(block(in_planes=self.in_planes, planes=planes, stride=s))
            self.in_planes = planes
        return nn.Sequential(*layer)

    def forward(self, x):
        out_std, out_param = self.conv1(x)
        out_param = self.bn_param(out_param)
        out_std = self.bn_std(out_std)
        out = torch.max(out_param, out_std)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.adaptive_avg_pool2d(out, (1,1))
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

def MakeAdditiveHybridGaborResNet():
    return AdditiveHybridGaborResNet([
        BasicBlock,
        BasicBlock,
        BasicBlock,
        BasicBlock,
    ], [2, 2, 2, 2])