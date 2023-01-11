import torch
import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo


__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']
model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


def initial(rank=1, tensor_shape=None):
    multiplier = (1 / rank) ** (1/3)
    return torch.ones(tensor_shape)*multiplier + 0.08*nn.init.kaiming_normal_(torch.empty(tensor_shape))


class MaskBlock(nn.Module):
    def __init__(self, shape, rank):
        super(MaskBlock, self).__init__()

        self.rank = rank
        self.maskcconv = nn.parameter.Parameter(initial(self.rank, (shape[0], self.rank)), requires_grad=True)
        self.maskhconv = nn.parameter.Parameter(initial(self.rank, (shape[1], self.rank)), requires_grad=True)
        self.maskwconv = nn.parameter.Parameter(initial(self.rank, (shape[2], self.rank)), requires_grad=True)
        self.shape = shape

    def forward(self, out):
        # print(out.shape)
        # out_f = torch.mul(out, self.maskcconv[:, 0].view(1,-1,1,1))
        # out_f = torch.mul(out_f, self.maskhconv[:, 0].view(1,1,-1,1))
        # out_f = torch.mul(out_f, self.maskwconv[:, 0].view(1,1,1,-1))
        # for r in range(1, self.rank):
        #     f = torch.mul(out, self.maskcconv[:, r].view(1,-1,1,1))
        #     f = torch.mul(f, self.maskhconv[:, r].view(1,1,-1,1))
        #     f = torch.mul(f, self.maskwconv[:, r].view(1,1,1,-1))
        #     out_f = torch.add(out_f, f)
        # out = out_f

        out_f = None
        for r in range(0, self.rank):
            # f = torch.mul(out, self.maskcconv[:, r].view(1, -1, 1, 1))
            # f = torch.mul(f, self.maskhconv[:, r].view(1, 1, -1, 1))
            # f = torch.mul(f, self.maskwconv[:, r].view(1, 1, 1, -1))

            f = torch.mul(self.maskcconv[:, r].view(1, -1, 1, 1), self.maskhconv[:, r].view(1, 1, -1, 1))
            f = torch.mul(f, self.maskwconv[:, r].view(1, 1, 1, -1))
            f = torch.mul(out, f)
            if out_f is None:
                out_f = f
            else:
                out_f = torch.add(out_f, f)

        return out_f


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
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


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, mask_shapes=None, first_shape=None, add_mask=False, mask_rank=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.add_mask = add_mask
        self.mask_rank = mask_rank

        if self.add_mask:
            if first_shape:
                self.mask1 = MaskBlock(first_shape, self.mask_rank)
            else:
                self.mask1 = MaskBlock((mask_shapes[0] // 4, mask_shapes[1], mask_shapes[2]), self.mask_rank)

            self.mask2 = MaskBlock((mask_shapes[0] // 4, mask_shapes[1], mask_shapes[2]), self.mask_rank)
            self.mask3 = MaskBlock(mask_shapes, self.mask_rank)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        if self.add_mask:
            out = self.mask1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        if self.add_mask:
            out = self.mask2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        if self.add_mask:
            out = self.mask3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, use_masks, mask_rank=1, num_classes=1000, input_channels=3):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], mask_shapes=[256, 56, 56], first_shape=[64, 56, 56], add_mask=use_masks[0], mask_rank=mask_rank)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, mask_shapes=[512, 28, 28], first_shape=[128, 56, 56], add_mask=use_masks[1], mask_rank=mask_rank)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, mask_shapes=[1024, 14, 14], first_shape=[256, 28, 28], add_mask=use_masks[2], mask_rank=mask_rank)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, mask_shapes=[2048, 7, 7], first_shape=[512, 14, 14], add_mask=use_masks[3], mask_rank=mask_rank)
        self.avgpool = nn.AvgPool2d(7, stride=2)

        self.dropout = nn.Dropout2d(p=0.5, inplace=True)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, mask_shapes=None, first_shape=None, add_mask=False, mask_rank=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, mask_shapes=mask_shapes, first_shape=first_shape, add_mask=add_mask, mask_rank=mask_rank))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, mask_shapes=mask_shapes, add_mask=add_mask, mask_rank=mask_rank))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = self.dropout(x)

        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def resnet14(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [1, 1, 1, 1], **kwargs)
    if pretrained:
        raise RuntimeError("No pretrained resnet-14.")
    return model


def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    return model


def resnet34(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
    return model


def resnet50(num_classes, pretrained=True, use_masks=None, mask_rank=1, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        num_classes (int): Number of classes to predict
        use_masks (list): Whether to use mask in every layer
        mask_rank (int): Rank of the mask layer
    """
    if use_masks is None:
        use_masks = [False, False, False, True]
    model = ResNet(Bottleneck, [3, 4, 6, 3], use_masks, mask_rank=mask_rank, **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']), strict=False)
    model.fc = nn.Linear(2048, num_classes)
    return model


def resnet101(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet101']))
    return model


def resnet152(pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet152']))
    return model
