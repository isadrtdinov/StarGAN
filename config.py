from dataclasses import dataclass


@dataclass
class Params:
    # general config
    project: str = 'stargan'
    random_seed: int = 1010101
    img_size: int = 64
    crop_size: int = 178
    img_channels: int = 3
    attributes: tuple = ('Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Male', 'Young')

    # architecture params
    conv_channels: int = 32
    generator_blocks: int = 2
    critic_blocks: int = 4
    kernel: int = 5
    neg_slope: float = 0.1

    # training params
    num_workers: int = 8
    batch_size: int = 64
    num_epochs: int = 30
    generator_lr: float = 3e-4
    critic_lr: float = 1e-4

    # loss params
    lambda_clf: float = 1.0
    lambda_rec: float = 10.0

    # logging & checkpoints params
    example_ids: tuple = (2, 11, 21, 27, 33)
    example_domains: int = 7
    log_steps: int = 50
    valid_epochs: int = 2
    checkpoint_epochs: int = 5
    checkpoint_template: str = 'StarGAN{}.pt'
