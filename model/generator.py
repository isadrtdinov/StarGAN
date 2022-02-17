import math
import torch
from torch import nn
from model.layers import DownsampleBlock, UpsampleBlock


class Generator(nn.Module):
    """
    U-Net like generator network
    """
    def __init__(self, img_channels=3, cond_channels=5, conv_channels=32,
                 num_blocks=2, kernel=5, neg_slope=0.1):
        super().__init__()
        block_channels = [conv_channels * 2 ** i for i in range(num_blocks)]

        self.input_block = nn.Sequential(
            nn.Conv2d(img_channels, conv_channels, kernel_size=kernel,
                      stride=1, padding=math.ceil((kernel - 1) / 2)),
            nn.InstanceNorm2d(conv_channels),
            nn.LeakyReLU(neg_slope)
        )

        self.downsample_blocks = nn.ModuleList([
            DownsampleBlock(channels, kernel, neg_slope) for channels in block_channels
        ])

        block_channels[-1] += cond_channels
        self.upsample_blocks = nn.ModuleList([
            UpsampleBlock(channels, kernel, neg_slope) for channels in block_channels[::-1]
        ])

        self.head = nn.Sequential(
            nn.Conv2d(conv_channels, img_channels, kernel_size=kernel,
                      stride=1, padding=math.ceil((kernel - 1) / 2)),
            nn.Tanh()
        )

    def forward(self, images, conds):
        features_list = [self.input_block(images)]
        outputs = features_list[-1]

        for block in self.downsample_blocks:
            features_list += [block(outputs)]
            outputs = features_list[-1]

        conds = conds.reshape((conds.shape[0], conds.shape[1], 1, 1)) \
            .repeat(1, 1, outputs.shape[2], outputs.shape[3]).to(torch.float)
        outputs = torch.cat([outputs, conds], dim=1)

        for block, features in zip(self.upsample_blocks, features_list[-2::-1]):
            outputs = block(outputs, features)

        return self.head(outputs)
