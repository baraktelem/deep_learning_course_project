import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.layers.BasicBlock import BasicBlock
from src.models.layers.HybridGaborLayer import HybridGaborLayer

class HybridGaborResNet(nn.Module):
    def __init__(self, blocks, num_blocks, num_classes=10):
        super(HybridGaborResNet, self).__init__()
        self.in_planes = 64
        self.conv1 = HybridGaborLayer(3, 64, conv_kernel_size=3, gabor_kernel_size=13, ratio=0.5, pad_mode='constant')
        self.bn1 = nn.BatchNorm2d(64)
        
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
        out = self.conv1(x)
        out = F.relu(self.bn1(out))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.adaptive_avg_pool2d(out, (1,1))
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

def MakeHybridGaborResNet18():
    return HybridGaborResNet([
        BasicBlock,
        BasicBlock,
        BasicBlock,
        BasicBlock
    ], [2, 2, 2, 2])