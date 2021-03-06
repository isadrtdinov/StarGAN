import torch


def get_trg_labels(src_labels):
    rand_perm = torch.randperm(src_labels.shape[0])
    trg_labels = src_labels[rand_perm]
    return trg_labels
