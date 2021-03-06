from dataclasses import dataclass


@dataclass
class Params:
    # general config
    project: str = 'stargan'
    img_size: int = 128
    img_channels: int = 3
    attributes: tuple[str] = ('Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Male', 'Young')

    # architecture params
    conv_channels: int = 32
    generator_blocks: int = 2
    critic_blocks: int = 4
    kernel: int = 5
    neg_slope: float = 0.1

    # training params
    batch_size: int = 64
    num_epochs: int = 10
    generator_lr: float = 1e-4
    critic_lr: float = 1e-4

    # loss params
    lambda_clf: float = 1.0
    lambda_rec: float = 10.0
