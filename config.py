from dataclasses import dataclass


@dataclass
class Params:
    # general config
    project: str = 'stargan'
    random_seed: int = 1010101
    img_size: int = 160
    crop_size: int = 178
    img_channels: int = 3
    attributes: tuple = ('Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Gray_Hair', 'Bald',
                         'Goatee', 'Mustache', 'No_Beard', 'Smiling', 'Male', 'Young')

    # architecture params
    conv_channels: int = 64
    generator_blocks: int = 3
    critic_blocks: int = 5
    kernel: int = 7
    neg_slope: float = 0.1

    # training params
    num_workers: int = 8
    batch_size: int = 16
    num_epochs: int = 30
    generator_lr: float = 1e-4
    critic_lr: float = 5e-4

    # loss params
    lambda_clf: float = 1.0
    lambda_rec: float = 10.0

    # logging & checkpoints params
    example_ids: tuple = (2161, 6464, 8350, 13154, 10009, 18619, 11614, 1135)
    example_domains: int = 11
    log_steps: int = 50
    valid_epochs: int = 1
    checkpoint_epochs: int = 1
    checkpoint_template: str = 'StarGAN{}.pt'
