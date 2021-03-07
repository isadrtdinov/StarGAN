import torch
import numpy as np
from scipy import linalg
from torch import nn
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision.models import inception_v3
from utils import permute_labels


class FrechetInceptionDistance(object):
    """
    Calculates Frechet Inception Distance
    (a.k.a FID-score) for StarGAN evaluation
    """
    def __init__(self, device, batch_size=64, eps=1e-6):
        self.device = device
        self.batch_size = batch_size
        self.eps = eps

        self.transform = T.Compose([
            lambda x: x * 0.5 + 0.5,
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),  # normalize as ImageNet picture
            lambda x: F.interpolate(x, size=(299, 299), mode='bilinear',  # rescale for Inception
                                    align_corners=False, recompute_scale_factor=False)
        ])

        self.inception = inception_v3(pretrained=True, progress=True)
        self.inception.fc = nn.Identity()  # a hack to retrieve latent features from Inception

    def get_latent_features(self, model, dataloader):
        real_features, fake_features = [], []

        for real_images, src_labels in dataloader:
            trg_labels = permute_labels(src_labels)
            fake_images = model.generate(real_images, trg_labels)

            real_images = self.transform(real_images.to(self.device))
            fake_images = self.transform(fake_images)

            real_features += [self.inception(real_images).detach().cpu()]
            fake_features += [self.inception(fake_images).detach().cpu()]

        real_features = torch.cat(real_features, dim=0)
        fake_features = torch.cat(fake_features, dim=0)
        return real_features, fake_features

    def calculate_statistics(self, features):
        num_objects, features_dim = features.shape
        mu = features.mean(dim=0)
        sigma = torch.zeros((features_dim, features_dim), dtype=torch.float)

        for i in range(0, num_objects, self.batch_size):
            batch = (features[i:i + self.batch_size] - mu).to(self.device)
            sigma += (batch.unsqueeze(1) * batch.unsqueeze(2)).mean(dim=0).cpu()

        sigma *= self.batch_size / (num_objects - 1)
        return mu, sigma

    def calculate_frechet_distance(self, mu1, mu2, sigma1, sigma2):
        mu1, mu2 = mu1.numpy(), mu2.numpy()
        sigma1, sigma2 = sigma1.numpy(), sigma2.numpy()

        diff = mu1 - mu2
        offset = np.eye(sigma1.shape[0]) * self.eps
        covmean, _ = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset), disp=False)
        if np.iscomplexobj(covmean):
            covmean = covmean.real

        distance = diff.dot(diff) + np.trace(sigma1) + \
                   np.trace(sigma2) - 2 * np.trace(covmean)
        return distance

    @torch.no_grad()
    def fid_score(self, model, dataloader):
        self.inception = self.inception.to(self.device)
        self.inception.eval()
        model.eval()

        real_features, fake_features = self.get_latent_features(model, dataloader)
        mu_real, sigma_real = self.calculate_statistics(real_features)
        mu_fake, sigma_fake = self.calculate_statistics(fake_features)
        distance = self.calculate_frechet_distance(mu_real, mu_fake, sigma_real, sigma_fake)

        self.inception = self.inception.cpu()
        return distance
