import math
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm as SpectralNorm


class BilinearInterpolation(nn.Module):
    """
    F.interpolate class wrapper
    """
    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode='bilinear',
                          align_corners=False, recompute_scale_factor=False)
        return x


class DownsampleBlock(nn.Module):
    """
    A U-Net downsample block, with one regular convolution,
    one strided convolution, instance norm and leaky ReLU activations
    """
    def __init__(self, block_channels, kernel=5, neg_slope=0.1):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(block_channels, block_channels, kernel_size=kernel,
                      stride=1, padding=math.ceil((kernel - 1) / 2)),
            nn.LeakyReLU(neg_slope),
            nn.Conv2d(block_channels, 2 * block_channels, kernel_size=kernel,
                      stride=2, padding=math.ceil((kernel - 2) / 2)),
            nn.InstanceNorm2d(2 * block_channels, affine=True),
            nn.LeakyReLU(neg_slope)
        )

    def forward(self, x):
        return self.layers(x)


class UpsampleBlock(nn.Module):
    """
    A U-Net upsample block, which interpolates input features,
    concatenates them with the respective downsample features
    and passes through a regular convolution
    """
    def __init__(self, block_channels, kernel=5, neg_slope=0.1):
        super().__init__()

        self.upsample = nn.Sequential(
            BilinearInterpolation(),
            nn.Conv2d(2 * block_channels, block_channels, kernel_size=kernel,
                      stride=1, padding=math.ceil((kernel - 1) / 2)),
            nn.Tanh()
        )

        self.conv = nn.Sequential(
            nn.Conv2d(2 * block_channels, block_channels, kernel_size=kernel,
                      stride=1, padding=math.ceil((kernel - 1) / 2)),
            nn.InstanceNorm2d(block_channels, affine=True),
            nn.LeakyReLU(neg_slope),
        )

    def forward(self, inputs, features):
        outputs = self.upsample(inputs)
        outputs = torch.cat([outputs, features], dim=1)
        return self.conv(outputs)


class CriticBlock(nn.Module):
    """
    A critic block with a strided convolution and spectral normalization
    """
    def __init__(self, in_channels, out_channels, kernel=5, neg_slope=0.1):
        super().__init__()

        self.conv = nn.Sequential(
            SpectralNorm(
                nn.Conv2d(in_channels, out_channels, kernel_size=kernel,
                          stride=2, padding=math.ceil((kernel - 1) / 2), bias=False)
            ),
            nn.LeakyReLU(neg_slope)
        )

    def forward(self, x):
        return self.conv(x)
