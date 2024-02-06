# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

###########################################################################################
# 2D MODELS AS USED In THE PAPER:                                                         #
# Clifford Neural Layers for PDE Modeling                                                 #
###########################################################################################
from typing import Callable, Union

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import Conv2d, GroupNorm, Conv3d

from cliffordlayers.nn.modules.cliffordconv import CliffordConv2d
from cliffordlayers.nn.modules.cliffordfourier import CliffordSpectralConv2d
from cliffordlayers.nn.modules.groupnorm import CliffordGroupNorm2d
from cliffordlayers.models.basic.custom_layers import CliffordConv2dScalarVectorEncoder, CliffordConv2dScalarVectorDecoder


class CliffordBasicBlock2d(nn.Module):
    """2D building block for Clifford ResNet architectures.

    Args:
        g (Union[tuple, list, torch.Tensor]): Signature of Clifford algebra.
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        activation (Callable, optional): Activation function. Defaults to F.gelu.
        kernel_size (int, optional): Kernel size of Clifford convolution. Defaults to 3.
        stride (int, optional): Stride of Clifford convolution. Defaults to 1.
        padding (int, optional): Padding of Clifford convolution. Defaults to 1.
        rotation (bool, optional): Wether to use rotational Clifford convolution. Defaults to False.
        norm (bool, optional): Wether to use Clifford (group) normalization. Defaults to False.
        num_groups (int, optional): Number of groups when using Clifford (group) normalization. Defaults to 1.
    """

    expansion: int = 1

    def __init__(
        self,
        g: Union[tuple, list, torch.Tensor],
        in_channels: int,
        out_channels: int,
        activation: Callable = F.gelu,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        rotation: bool = False,
        norm: bool = False,
        num_groups: int = 1,
    ) -> None:
        super().__init__()
        self.conv1 = CliffordConv2d(
            g,
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=True,
            rotation=rotation,
        )
        self.conv2 = CliffordConv2d(
            g,
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=True,
            rotation=rotation,
        )
        self.norm1 = CliffordGroupNorm2d(g, num_groups, in_channels) if norm else nn.Identity()
        self.norm2 = CliffordGroupNorm2d(g, num_groups, out_channels) if norm else nn.Identity()
        self.activation = activation

    def __repr__(self):
        return "CliffordBasicBlock2d"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(self.activation(self.norm1(x)))
        out = self.conv2(self.activation(self.norm2(out)))
        return out + x


class CliffordFourierBasicBlock2d(nn.Module):
    """2D building block for Clifford FNO architectures.

    Args:
        g (Union[tuple, list, torch.Tensor]): Signature of Clifford algebra.
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        activation (Callable, optional): Activation function. Defaults to F.gelu.
        kernel_size (int, optional): Kernel size of Clifford convolution. Defaults to 3.
        stride (int, optional): Stride of Clifford convolution. Defaults to 1.
        padding (int, optional): Padding of Clifford convolution. Defaults to 1.
        rotation (bool, optional): Wether to use rotational Clifford convolution. Defaults to False.
        norm (bool, optional): Wether to use Clifford (group) normalization. Defaults to False.
        num_groups (int, optional): Number of groups when using Clifford (group) normalization. Defaults to 1.
        modes1 (int, optional): Number of Fourier modes in the first dimension. Defaults to 16.
        modes2 (int, optional): Number of Fourier modes in the second dimension. Defaults to 16.
    """

    expansion: int = 1

    def __init__(
        self,
        g: Union[tuple, list, torch.Tensor],
        in_channels: int,
        out_channels: int,
        activation: Callable = F.gelu,
        kernel_size: int = 1,
        stride: int = 1,
        padding: int = 0,
        rotation: bool = False,
        norm: bool = False,
        num_groups: int = 1,
        modes1: int = 16,
        modes2: int = 16,
    ):
        super().__init__()
        self.fourier = CliffordSpectralConv2d(
            g,
            in_channels,
            out_channels,
            modes1=modes1,
            modes2=modes2,
        )
        self.conv = CliffordConv2d(
            g,
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=True,
            rotation=rotation,
        )
        self.norm = CliffordGroupNorm2d(g, num_groups, out_channels) if norm else nn.Identity()
        self.activation = activation

    def __repr__(self):
        return "CliffordFourierBasicBlock2d"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.fourier(x)
        x2 = self.conv(x)
        return self.activation(self.norm(x1 + x2))


class CliffordFluidNet2d(nn.Module):
    """2D building block for Clifford architectures for fluid mechanics (vector field+scalar field)
    with ResNet backbone network. The backbone networks follows these three steps:
        1. Clifford scalar+vector field encoding.
        2. Basic blocks as provided.
        3. Clifford scalar+vector field decoding.

    Args:
        g (Union[tuple, list, torch.Tensor]): Signature of Clifford algebra.
        block (nn.Module): Choice of basic blocks.
        num_blocks (list): List of basic blocks in each residual block.
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        activation (Callable, optional): Activation function. Defaults to F.gelu.
        rotation (bool, optional): Wether to use rotational Clifford convolution. Defaults to False.
        norm (bool, optional): Wether to use Clifford (group) normalization. Defaults to False.
        num_groups (int, optional): Number of groups when using Clifford (group) normalization. Defaults to 1.
    """

    # For periodic boundary conditions, set padding = 0.
    padding = 9

    def __init__(
        self,
        g: Union[tuple, list, torch.Tensor],
        block: nn.Module,
        num_blocks: list,
        in_channels: int,
        out_channels: int,
        hidden_channels: int,
        activation: Callable,
        rotation: False,
        norm: bool = False,
        num_groups: int = 1,
    ):
        super().__init__()

        self.activation = activation
        # Encoding and decoding layers
        self.encoder = CliffordConv2dScalarVectorEncoder(
            g,
            in_channels=in_channels,
            out_channels=hidden_channels,
            kernel_size=1,
            padding=0,
            rotation=rotation,
        )
        self.decoder = CliffordConv2dScalarVectorDecoder(
            g,
            in_channels=hidden_channels,
            out_channels=out_channels,
            kernel_size=1,
            padding=0,
            rotation=rotation,
        )

        # Residual blocks
        self.layers = nn.ModuleList(
            [
                self._make_basic_block(
                    g,
                    block,
                    hidden_channels,
                    num_blocks[i],
                    activation=activation,
                    rotation=rotation,
                    norm=norm,
                    num_groups=num_groups,
                )
                for i in range(len(num_blocks))
            ]
        )

    def _make_basic_block(
        self,
        g,
        block: nn.Module,
        hidden_channels: int,
        num_blocks: int,
        activation: Callable,
        rotation: bool,
        norm: bool,
        num_groups: int,
    ) -> nn.Sequential:
        blocks = []
        for _ in range(num_blocks):
            blocks.append(
                block(
                    g,
                    hidden_channels,
                    hidden_channels,
                    activation=activation,
                    rotation=rotation,
                    norm=norm,
                    num_groups=num_groups,
                )
            )
        return nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.dim() == 5

        # Encoding layer
        x = self.encoder(self.activation(x))

        # Embed for non-periodic boundaries
        if self.padding > 0:
            B_dim, C_dim, *D_dims, I_dim = range(len(x.shape))
            x = x.permute(B_dim, I_dim, C_dim, *D_dims)
            x = F.pad(x, [0, self.padding, 0, self.padding])
            B_dim, I_dim, C_dim, *D_dims = range(len(x.shape))
            x = x.permute(B_dim, C_dim, *D_dims, I_dim)

        # Apply residual layers
        for layer in self.layers:
            x = layer(x)

        # Decoding layer
        if self.padding > 0:
            B_dim, C_dim, *D_dims, I_dim = range(len(x.shape))
            x = x.permute(B_dim, I_dim, C_dim, *D_dims)
            x = x[..., : -self.padding, : -self.padding]
            B_dim, I_dim, C_dim, *D_dims = range(len(x.shape))
            x = x.permute(B_dim, C_dim, *D_dims, I_dim)

        # Output layer
        x = self.decoder(x)
        return x


"""
Builds ResNet18 from scratch using PyTorch.
This does not build generalized blocks for all ResNets, just for ResNet18.
Paper => Deep Residual Learning for Image Recognition.
Link => https://arxiv.org/pdf/1512.03385v1.pdf
"""

import torch.nn as nn
import torch

from torch import Tensor
from typing import Type


class BasicBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        expansion: int = 1,
        downsample: nn.Module = None,
    ) -> None:
        super(BasicBlock, self).__init__()
        # Multiplicative factor for the subsequent conv2d layer's output channels.
        # It is 1 for ResNet18 and ResNet34.
        self.expansion = expansion
        self.downsample = downsample
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.gelu = nn.GELU()
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels * self.expansion,
            kernel_size=3,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_channels * self.expansion)

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class BasicBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        expansion: int = 1,
        downsample: nn.Module = None,
    ) -> None:
        super(BasicBlock, self).__init__()
        # Multiplicative factor for the subsequent conv2d layer's output channels.
        # It is 1 for ResNet18 and ResNet34.
        self.expansion = expansion
        self.downsample = downsample
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.gelu = nn.GELU()
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels * self.expansion,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_channels * self.expansion)

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)

        out = self.bn1(out)
        out = self.gelu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.gelu(out)
        return out
    
class ResNet(nn.Module):
    def __init__(
        self,
        img_channels: int,
        out_channels:int,
        num_layers: int,
        block: Type[BasicBlock],
    ) -> None:
        super(ResNet, self).__init__()
        if num_layers == 18:
            # The following `layers` list defines the number of `BasicBlock`
            # to use to build the network and how many basic blocks to stack
            # together.
            layers = [2, 2, 2, 2]
            self.expansion = 1

        self.in_channels = 128
        # All ResNets (18 to 152) contain a Conv2d => BN => ReLU for the first
        # three layers. Here, kernel size is 7.
        self.conv1 = nn.Conv2d(
            in_channels=img_channels,
            out_channels=self.in_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.relu = nn.ReLU()

        self.layer1 = self._make_layer(block, 128, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=1)
        self.layer3 = self._make_layer(block, 128, layers[2], stride=1)
        self.layer4 = self._make_layer(block, 128, layers[3], stride=1)

        self.final = nn.Conv2d(
            in_channels=128,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )


    def _make_layer(
        self, block: Type[BasicBlock], out_channels: int, blocks: int, stride: int = 1
    ) -> nn.Sequential:
        downsample = None
        if stride != 1:
            """
            This should pass from `layer2` to `layer4` or
            when building ResNets50 and above. Section 3.3 of the paper
            Deep Residual Learning for Image Recognition
            (https://arxiv.org/pdf/1512.03385v1.pdf).
            """
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.in_channels,
                    out_channels * self.expansion,
                    kernel_size=3,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels * self.expansion),
            )
        layers = []
        layers.append(
            block(self.in_channels, out_channels, stride, self.expansion, downsample)
        )
        self.in_channels = out_channels * self.expansion

        for i in range(1, blocks):
            layers.append(
                block(self.in_channels, out_channels, expansion=self.expansion)
            )
        return nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        #x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        # The spatial dimension of the final layer's feature
        # map should be (7, 7) for all ResNets.
        x = self.final(x)
        return x

