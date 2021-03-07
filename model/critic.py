import math
from torch import nn
from torch.nn.utils import spectral_norm as SpectralNorm
from model.layers import CriticBlock



class Critic(nn.Module):
    """
    Critic network with one patch adversarial head and one classification head
    """
    def __init__(self, img_size=128, img_channels=3, cond_channels=5,
                 conv_channels=32, num_blocks=4, kernel=5, neg_slope=0.1):
        super().__init__()
        block_channels = [conv_channels * 2 ** i for i in range(num_blocks - 1)]

        self.blocks = [CriticBlock(img_channels, conv_channels, kernel, neg_slope)]
        self.blocks += [
            CriticBlock(channels, 2 * channels, kernel, neg_slope)
            for channels in block_channels
        ]
        self.blocks = nn.Sequential(*self.blocks)

        output_channels = conv_channels * 2 ** (num_blocks - 1)
        output_size = img_size // 2 ** num_blocks
        self.adv_head = SpectralNorm(nn.Conv2d(output_channels, 1, kernel_size=kernel,
                                               stride=1, padding=math.ceil((kernel - 1) / 2)))
        self.clf_head = SpectralNorm(nn.Conv2d(output_channels, cond_channels,
                                               kernel_size=output_size))

    def forward(self, images):
        outputs = self.blocks(images)
        adv_logits = self.adv_head(outputs)
        clf_logits = self.clf_head(outputs).squeeze()
        return adv_logits, clf_logits
