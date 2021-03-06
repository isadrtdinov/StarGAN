import torch
from torchvision import datasets


class CelebA(datasets.CelebA):
    """
    CelebA dataset modification restricted to the selected attributes
    """
    def __init__(self, root, attributes, split='train', target_type='attr', transform=None,
                 target_transform=None, download=False):
        super().__init__(root, split, target_type, transform, target_transform, download)
        index2attr = {i: attr for i, attr in enumerate(self.attr_names)}
        attr2index = {attr: i for i, attr in index2attr.items()}
        self.label_indices = [attr2index[attr] for attr in attributes]
        self.label_indices = torch.tensor(self.label_indices)

    def __getitem__(self, item):
        image, attrs = super().__getitem__(item)
        return image, attrs[self.label_indices]
