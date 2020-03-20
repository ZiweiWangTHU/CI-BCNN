import torch.nn as nn
import math
from .binarized_modules import BinarizeLinear, BinarizeConv2d

__all__ = ['resnet_binary']


def Binaryconv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return BinarizeConv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                          padding=1, bias=False)


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def init_model(model):
    for m in model.modules():
        if isinstance(m, BinarizeConv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, do_bntan=True):
        super(BasicBlock, self).__init__()

        self.conv1 = Binaryconv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.tanh1 = nn.Hardtanh(inplace=True)
        self.conv2 = Binaryconv3x3(planes, planes)
        self.tanh2 = nn.Hardtanh(inplace=True)
        self.bn2 = nn.BatchNorm2d(planes)

        self.downsample = downsample
        self.do_bntan = do_bntan
        self.stride = stride
        self.feature = []

    def forward(self, x):
        self.feature = []
        residual = x.clone()

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.tanh1(out)

        out = self.conv2(out)

        if self.downsample is not None:
            residual = self.downsample(residual)

        out += residual
        if self.do_bntan:
            out = self.bn2(out)
            out = self.tanh2(out)

        self.feature.append(self.conv1.feature)
        self.feature.append(self.conv2.feature)
        if self.downsample is not None:
            self.feature.append(self.downsample[0].feature)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = BinarizeConv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = BinarizeConv2d(planes, planes, kernel_size=3, stride=stride,
                                    padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = BinarizeConv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.tanh = nn.Hardtanh(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        import pdb
        pdb.set_trace()
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.tanh(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.tanh(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        if self.do_bntan:
            out = self.bn2(out)
            out = self.tanh2(out)

        return out


class ResNet(nn.Module):

    def __init__(self):
        super(ResNet, self).__init__()
        self.feature = []

    def _make_layer(self, block, planes, blocks, stride=1, do_bntan=True):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                BinarizeConv2d(self.inplanes, planes * block.expansion,
                               kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks - 1):
            layers.append(block(self.inplanes, planes))
        layers.append(block(self.inplanes, planes, do_bntan=do_bntan))
        return nn.Sequential(*layers)

    def forward(self, x):
        self.feature = []
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.bn1(x)
        x = self.tanh1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.bn2(x)
        x = self.tanh2(x)
        x = self.fc(x)
        x = self.bn3(x)
        x = self.logsoftmax(x)

        self.feature.extend(self.layer1[0].feature)
        self.feature.extend(self.layer1[1].feature)
        self.feature.extend(self.layer2[0].feature)
        self.feature.extend(self.layer2[1].feature)
        self.feature.extend(self.layer3[0].feature)
        self.feature.extend(self.layer3[1].feature)

        return x


class ResNet_cifar10(ResNet):
    def __init__(self, num_classes=10,
                 block=BasicBlock, depth=18):
        super(ResNet_cifar10, self).__init__()
        self.inflate = 5
        self.inplanes = 16 * self.inflate
        n = int((depth - 2) / 6)
        self.conv1 = nn.Conv2d(3, 16 * self.inflate, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.maxpool = lambda x: x
        self.bn1 = nn.BatchNorm2d(16 * self.inflate)
        self.tanh1 = nn.Hardtanh(inplace=True)
        self.layer1 = self._make_layer(block, 16 * self.inflate, n)
        self.layer2 = self._make_layer(block, 32 * self.inflate, n, stride=2)
        self.layer3 = self._make_layer(block, 64 * self.inflate, n, stride=2, do_bntan=False)
        self.layer4 = lambda x: x
        self.avgpool = nn.AvgPool2d(8)
        self.bn2 = nn.BatchNorm1d(64 * self.inflate)
        self.tanh2 = nn.Hardtanh(inplace=True)
        self.fc = BinarizeLinear(64 * self.inflate, num_classes)
        self.bn3 = nn.BatchNorm1d(10)
        self.logsoftmax = nn.LogSoftmax()

        init_model(self)

        self.regime = {
            0: {'optimizer': 'Adam', 'lr': 5e-3},
            101: {'lr': 1e-3},
            142: {'lr': 5e-4},
            184: {'lr': 1e-4},
            220: {'lr': 1e-5}
        }

    def set(self, xx, yy, inf_state):
        n = 0
        for m in self.modules():
            if isinstance(m, BinarizeConv2d):
                if n > len(inf_state) - 1:
                    break
                m.set(xx[n], yy[n], inf_state[n])
                n += 1


def resnet_binary():
    num_classes = 10
    depth = 18
    return ResNet_cifar10(num_classes=num_classes,
                          block=BasicBlock, depth=depth)

