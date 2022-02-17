import torch
import os
import wandb
import torchvision.transforms as T
from model.stargan import StarGAN
from model.fid import FrechetInceptionDistance
from dataset import CelebA
from config import Params
import utils


params = Params()
utils.set_random_seed(params.random_seed)
wandb.init(project=params.project, config=params.__dict__)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

transform = T.Compose([
    T.RandomHorizontalFlip(0.5),
    T.CenterCrop(params.crop_size),
    T.Resize((params.img_size, params.img_size)),
    T.ToTensor(),
    T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

train_dataset = CelebA('celeba', attributes=params.attributes, target_type='attr',
                       split='train', transform=transform, download=False)
test_dataset = CelebA('celeba', attributes=params.attributes, target_type='attr',
                      split='test', transform=transform, download=False)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=params.batch_size,
                                           shuffle=True, drop_last=True, pin_memory=True,
                                           num_workers=params.num_workers)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=params.batch_size,
                                          shuffle=True, drop_last=True, pin_memory=True,
                                          num_workers=params.num_workers)

model = StarGAN(params, device)
wandb.watch([model.net_G, model.net_D])
fid = FrechetInceptionDistance(device, batch_size=params.batch_size)

orig_images, example_images, example_labels = \
    utils.generate_examples(test_dataset, example_ids=params.example_ids,
                            example_domains=params.example_domains)

model.train()
for epoch in range(1, params.num_epochs + 1):
    for real_images, src_labels in train_loader:
        trg_labels = utils.permute_labels(src_labels)
        model.train_D(real_images, src_labels, trg_labels)
        model.train_G(real_images, src_labels, trg_labels)

        if model.train_step % params.log_steps == 0:
            wandb.log(model.metrics)

        if model.train_step % params.example_steps == 0:
            model.eval()
            generated_images = model.generate(example_images, example_labels).cpu()
            example_grid = utils.process_examples(orig_images, generated_images,
                                                  params.example_domains)
            model.train()

            wandb.log({
                'example': wandb.Image(example_grid),
                'train step': model.train_step,
            })

        if model.train_step % params.fid_steps == 0:
            model.eval()
            fid_score = fid.fid_score(model, test_loader)
            model.train()

            wandb.log({
                'FID score': fid_score,
                'train step': model.train_step,
            })

        if model.train_step % params.checkpoint_steps == 0:
            checkpoint_file = params.checkpoint_template.format(model.train_step)
            model.save_checkpoint(os.path.join(params.checkpoint_root, checkpoint_file))
