""" 
  ┌────────────────────────────────────────────────────┐
  │ Adaptation for resnet_1D                           │
  └────────────────────────────────────────────────────┘
 """

#* Import

import torch
import torch.nn as nn


""" 
 *┌──────────────────┐
 !│ Custom 1D Resnet │
 *└──────────────────┘
 """

class Block1D(nn.Module):
    """
    1D Residual Block for ResNet1D.

    This block consists of a series of convolutional and batch normalization layers,
    along with a skip connection.

    Parameters:
        - in_channels (int): Number of input channels.
        - intermediate_channels (int): Number of intermediate channels for convolutional layers.
        - identity_downsample (nn.Sequential): Skip connection to adjust dimensions if necessary.
        - stride (int): Stride for convolutional layers.
    """
    expansion = 4

    def __init__(self, in_channels:int, intermediate_channels:int, identity_downsample:nn.Sequential=None, stride:int=1):
        super(Block1D, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, intermediate_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm1d(intermediate_channels)
        self.conv2 = nn.Conv1d(intermediate_channels, intermediate_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(intermediate_channels)
        self.conv3 = nn.Conv1d(intermediate_channels, intermediate_channels * self.expansion, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm1d(intermediate_channels * self.expansion)
        self.relu = nn.ReLU()
        self.identity_downsample = identity_downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

        if self.identity_downsample is not None:
            identity = self.identity_downsample(identity)

        out += identity
        out = self.relu(out)
        return out

class ResNet1D(nn.Module):
    """
    1D Convolutional Neural Network (CNN) based on ResNet for sequence data.

    Parameters:
        - block (nn.Module): Type of residual block to use (e.g., Block1D).
        - layers (list): List specifying the number of residual blocks in each layer.
        - input_channels (int): Number of input channels.
        - num_classes (int): Number of output classes.
    """
    def __init__(self, block:nn.Module, layers:list, input_channels:int, num_classes:int):
        super(ResNet1D, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv1d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU()
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block:nn.Module, intermediate_channels:int, num_blocks:int, stride:int):
        """
        Creates a layer of residual blocks.

        Parameters:
            - block (nn.Module): Type of residual block to use (e.g., Block1D).
            - intermediate_channels (int): Number of intermediate channels for convolutional layers.
            - num_blocks (int): Number of residual blocks in the layer.
            - stride (int): Stride for convolutional layers.
        """
        identity_downsample = None
        layers = []

        if stride != 1 or self.in_channels != intermediate_channels * block.expansion:
            identity_downsample = nn.Sequential(
                nn.Conv1d(self.in_channels, intermediate_channels * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(intermediate_channels * block.expansion)
            )

        layers.append(block(self.in_channels, intermediate_channels, identity_downsample, stride))
        self.in_channels = intermediate_channels * block.expansion

        for _ in range(1, num_blocks):
            layers.append(block(self.in_channels, intermediate_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


def ResNet18_1D(input_channels:int=1, num_classes:int=2):
    """
    Constructs a ResNet-18 model for 1D sequence data classification.

    Parameters:
        - input_channels (int, optional): Number of input channels (default: 1).
        - num_classes (int, optional): Number of output classes (default: 2).

    Returns:
        - ResNet1D: ResNet-18 model configured for 1D sequence data.
    """
    return ResNet1D(Block1D, [2, 2, 2, 2], input_channels, num_classes)

def ResNet34_1D(input_channels:int=1, num_classes:int=2):
    """
    Constructs a ResNet-34 model for 1D sequence data classification.

    Parameters:
        - input_channels (int, optional): Number of input channels (default: 1).
        - num_classes (int, optional): Number of output classes (default: 2).

    Returns:
        - ResNet1D: ResNet-34 model configured for 1D sequence data.
    """
    return ResNet1D(Block1D, [3, 4, 6, 3], input_channels, num_classes)

def ResNet50_1D(input_channels:int=1, num_classes:int=2):
    """
    Constructs a ResNet-50 model for 1D sequence data classification.

    Parameters:
        - input_channels (int, optional): Number of input channels (default: 1).
        - num_classes (int, optional): Number of output classes (default: 2).

    Returns:
        - ResNet1D: ResNet-50 model configured for 1D sequence data.
    """
    return ResNet1D(Block1D, [3, 4, 6, 3], input_channels, num_classes)

def ResNet101_1D(input_channels:int=1, num_classes:int=2):
    """
    Constructs a ResNet-101 model for 1D sequence data classification.

    Parameters:
        - input_channels (int, optional): Number of input channels (default: 1).
        - num_classes (int, optional): Number of output classes (default: 2).

    Returns:
        - ResNet1D: ResNet-101 model configured for 1D sequence data.
    """
    return ResNet1D(Block1D, [3, 4, 23, 3], input_channels, num_classes)

def ResNet152_1D(input_channels:int=1, num_classes:int=2):
    """
    Constructs a ResNet-152 model for 1D sequence data classification.

    Parameters:
        - input_channels (int, optional): Number of input channels (default: 1).
        - num_classes (int, optional): Number of output classes (default: 2).

    Returns:
        - ResNet1D: ResNet-152 model configured for 1D sequence data.
    """
    return ResNet1D(Block1D, [3, 8, 36, 3], input_channels, num_classes)