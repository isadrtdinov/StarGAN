import random
import numpy as np
import torch
from torchvision.utils import make_grid


def set_random_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def permute_labels(src_labels):
    rand_perm = torch.randperm(src_labels.shape[0])
    trg_labels = src_labels[rand_perm]
    return trg_labels


@torch.no_grad()
def generate_examples(dataset, example_ids, example_domains):
    orig_images, example_images, example_labels = [], [], []

    for example_id in example_ids:
        image, attrs = dataset[example_id]
        orig_images += [image]
        example_images += [image.unsqueeze(0).repeat(example_domains, 1, 1, 1)]
        labels = attrs.unsqueeze(0).repeat(example_domains, 1)
        labels[:3, :3] = torch.eye(3)
        labels[3:5, 3] = torch.arange(2)
        labels[5:, 4] = torch.arange(2)
        example_labels += [labels]

    orig_images = torch.stack(orig_images, dim=0)
    example_images = torch.cat(example_images, dim=0)
    example_labels = torch.cat(example_labels, dim=0)

    return orig_images, example_images, example_labels


@torch.no_grad()
def process_examples(orig_images, generated_images, example_domains):
    shape = generated_images.shape
    orig_images = orig_images.unsqueeze(1)
    generated_images = generated_images.reshape(-1, example_domains,
                                                shape[1], shape[2], shape[3])

    images_grid = torch.cat([orig_images, generated_images], dim=1)
    images_grid = images_grid.reshape(-1, shape[1], shape[2], shape[3])
    images_grid = make_grid(images_grid, nrow=example_domains + 1)
    images_grid = (images_grid * 0.5 + 0.5).permute(1, 2, 0).numpy()

    return images_grid
