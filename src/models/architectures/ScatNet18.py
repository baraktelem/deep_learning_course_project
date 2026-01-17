import torch
import torch.nn as nn
import torch.nn.functional as F
from kymatio.torch import Scattering2D

from src.models.layers.BasicBlock import BasicBlock


class ScatResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, L=8):
        super(ScatResNet, self).__init__()
        self.in_planes = 64
        self.L = L
        self.scat_channels = (1 + L) * 3
        self.conv1 = nn.Conv2d(3, 64 - self.scat_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.scat1 = Scattering2D(J=1, shape=(32, 32), L=L, max_order=2, backend='torch')
        self.deconv1 = nn.ConvTranspose2d(in_channels=self.scat_channels, out_channels=self.scat_channels, groups=self.scat_channels, kernel_size=2, stride=2)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layer = []
        for s in strides:
            layer.append(block(self.in_planes, planes, s))
            self.in_planes = planes
        return nn.Sequential(*layer)

    def forward(self, x):
        out = self.conv1(x)
        scat = self.scat1(x)
        scat = scat.view(scat.size(0), -1, 16,16)
        scat = self.deconv1(scat)
        out = torch.cat((out, scat), dim=1)
        out = F.relu(self.bn1(out))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

def MakeScatResNet18(L, num_classes=10):
    return ScatResNet(BasicBlock, [2, 2, 2, 2], L=L, num_classes=num_classes)