import torch
from torch import nn
from model.generator import Generator
from model.critic import Critic
from collections import defaultdict


class StarGAN(object):
    """
    Main StarGAN class, which implements
    initialization and training of the models
    """
    def __init__(self, params, device):
        self.params = params
        self.device = device

        self.net_G = Generator(params.img_channels, len(params.attributes),
                               params.conv_channels, params.generator_blocks,
                               params.kernel, params.neg_slope).to(device)
        self.net_D = Critic(params.img_size, params.img_channels, len(params.attributes),
                            params.conv_channels, params.critic_blocks,
                            params.kernel, params.neg_slope).to(device)

        self.optim_G = torch.optim.Adam(self.net_G.parameters(), params.generator_lr)
        self.optim_D = torch.optim.Adam(self.net_D.parameters(), params.critic_lr)

        self.bce_loss = nn.BCEWithLogitsLoss().to(device)
        self.l1_loss = nn.L1Loss().to(device)
        self.metrics = defaultdict(float)
        self.train_step = 0

    def save_checkpoint(self, file):
        torch.save({
            'net_G': self.net_G.state_dict(),
            'net_D': self.net_D.state_dict(),
            'optim_G': self.optim_G.state_dict(),
            'optim_D': self.optim_D.state_dict(),
            'metrics': self.metrics,
            'train_step': self.train_step
        }, file)

    def load_checkpoint(self, file):
        checkpoint = torch.load(file)
        self.net_G.load_state_dict(checkpoint['net_G'])
        self.net_D.load_state_dict(checkpoint['net_D'])
        self.optim_G.load_state_dict(checkpoint['optim_G'])
        self.optim_D.load_state_dict(checkpoint['optim_D'])
        self.metrics = checkpoint['metrics']
        self.train_step = checkpoint['train_step']

    def train(self):
        self.net_G.train()
        self.net_D.train()

    def eval(self):
        self.net_G.eval()
        self.net_D.eval()

    def clf_loss(self, clf_logits, labels):
        return self.bce_loss(clf_logits, labels.to(torch.float))

    def rec_loss(self, real_images, reconstructed_images):
        return self.l1_loss(real_images, reconstructed_images)

    def generate(self, images, trg_labels):
        images = images.to(self.device)
        trg_labels = trg_labels.to(self.device)
        return self.net_G(images, trg_labels)

    def train_D(self, real_images, src_labels, trg_labels):
        self.optim_D.zero_grad()
        real_images = real_images.to(self.device)
        src_labels = src_labels.to(self.device)
        trg_labels = trg_labels.to(self.device)

        with torch.no_grad():
            fake_images = self.net_G(real_images, trg_labels)

        real_adv_logits, real_clf_logits = self.net_D(real_images)
        fake_adv_logits, _ = self.net_D(fake_images)

        d_adv_loss = fake_adv_logits.mean() - real_adv_logits.mean()
        d_clf_loss = self.clf_loss(real_clf_logits, src_labels)
        d_loss = d_adv_loss + self.params.lambda_clf * d_clf_loss
        d_loss.backward()
        self.optim_D.step()

        self.metrics['critic adv loss'] = d_adv_loss.item()
        self.metrics['critic clf loss'] = d_clf_loss.item()
        self.metrics['critic total loss'] = d_loss.item()

    def train_G(self, real_images, src_labels, trg_labels):
        self.optim_G.zero_grad()
        real_images = real_images.to(self.device)
        src_labels = src_labels.to(self.device)
        trg_labels = trg_labels.to(self.device)

        fake_images = self.net_G(real_images, trg_labels)
        reconstructed_images = self.net_G(fake_images, src_labels)
        fake_adv_logits, fake_clf_logits = self.net_D(fake_images)

        g_adv_loss = -fake_adv_logits.mean()
        g_clf_loss = self.clf_loss(fake_clf_logits, trg_labels)
        g_rec_loss = self.rec_loss(real_images, reconstructed_images)
        g_loss = g_adv_loss + self.params.lambda_clf * g_clf_loss + \
                              self.params.lambda_rec * g_rec_loss
        g_loss.backward()
        self.optim_G.step()

        self.metrics['generator adv loss'] = g_adv_loss.item()
        self.metrics['generator clf loss'] = g_clf_loss.item()
        self.metrics['generator rec loss'] = g_rec_loss.item()
        self.metrics['generator total loss'] = g_loss.item()
        self.metrics['train step'] += 1
        self.train_step += 1
