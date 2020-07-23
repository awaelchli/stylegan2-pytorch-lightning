import argparse
import math
import random
import torch
import wandb
from pytorch_lightning import LightningModule, Trainer, seed_everything
from pytorch_lightning.loggers import WandbLogger
from torch import autograd, optim
from torch.nn import functional as F
from torch.utils import data
from torchvision import transforms, utils

from model import Generator, Discriminator
from dataset import MultiResolutionDataset
from non_leaking import augment


def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag


def accumulate(model1, model2, decay=0.999):
    par1 = dict(model1.named_parameters())
    par2 = dict(model2.named_parameters())

    for k in par1.keys():
        par1[k].data.mul_(decay).add_(par2[k].data, alpha=1 - decay)


def sample_data(loader):
    while True:
        for batch in loader:
            yield batch


def d_logistic_loss(real_pred, fake_pred):
    real_loss = F.softplus(-real_pred)
    fake_loss = F.softplus(fake_pred)

    return real_loss.mean() + fake_loss.mean()


def d_r1_loss(real_pred, real_img):
    grad_real, = autograd.grad(
        outputs=real_pred.sum(), inputs=real_img, create_graph=True
    )
    grad_penalty = grad_real.pow(2).reshape(grad_real.shape[0], -1).sum(1).mean()

    return grad_penalty


def g_nonsaturating_loss(fake_pred):
    loss = F.softplus(-fake_pred).mean()

    return loss


def g_path_regularize(fake_img, latents, mean_path_length, decay=0.01):
    noise = torch.randn_like(fake_img) / math.sqrt(
        fake_img.shape[2] * fake_img.shape[3]
    )
    grad, = autograd.grad(
        outputs=(fake_img * noise).sum(), inputs=latents, create_graph=True
    )
    path_lengths = torch.sqrt(grad.pow(2).sum(2).mean(1))

    path_mean = mean_path_length + decay * (path_lengths.mean() - mean_path_length)

    path_penalty = (path_lengths - path_mean).pow(2).mean()

    return path_penalty, path_mean.detach(), path_lengths


def make_noise(batch, latent_dim, n_noise, device):
    if n_noise == 1:
        return torch.randn(batch, latent_dim, device=device)

    noises = torch.randn(n_noise, batch, latent_dim, device=device).unbind(0)

    return noises


def mixing_noise(batch, latent_dim, prob, device):
    if prob > 0 and random.random() < prob:
        return make_noise(batch, latent_dim, 2, device)

    else:
        return [make_noise(batch, latent_dim, 1, device)]


def set_grad_none(model, targets):
    for n, p in model.named_parameters():
        if n in targets:
            p.grad = None


class StyleGAN2(LightningModule):

    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters()
        self.hparams = hparams
        self.hparams.latent = 512
        self.hparams.n_mlp = 8
        self.generator, self.g_ema = self.init_generator()
        self.discriminator = self.init_discriminator()

        self.mean_path_length = 0
        self.accum = 0.5 ** (32 / (10 * 1000))
        self.ada_aug_p = hparams.augment_p if hparams.augment_p > 0 else 0.0
        self.ada_aug_step = hparams.ada_target / hparams.ada_length
        self.register_buffer('ada_augment', torch.zeros(2))
        self.register_buffer('sample_z', torch.randn(self.hparams.n_sample, self.hparams.latent))

    def configure_optimizers(self):
        args = self.hparams
        g_reg_ratio = args.g_reg_every / (args.g_reg_every + 1)
        d_reg_ratio = args.d_reg_every / (args.d_reg_every + 1)
        g_optim = optim.Adam(
            self.generator.parameters(),
            lr=args.lr * g_reg_ratio,
            betas=(0 ** g_reg_ratio, 0.99 ** g_reg_ratio),
        )
        d_optim = optim.Adam(
            self.discriminator.parameters(),
            lr=args.lr * d_reg_ratio,
            betas=(0 ** d_reg_ratio, 0.99 ** d_reg_ratio),
        )
        return d_optim, g_optim

    def init_discriminator(self):
        args = self.hparams
        discriminator = Discriminator(
            args.size, channel_multiplier=args.channel_multiplier
        )
        return discriminator

    def init_generator(self):
        args = self.hparams
        generator = Generator(
            args.size, args.latent, args.n_mlp, channel_multiplier=args.channel_multiplier
        )
        g_ema = Generator(
            args.size, args.latent, args.n_mlp, channel_multiplier=args.channel_multiplier
        )
        g_ema.eval()
        accumulate(g_ema, generator, 0)
        return generator, g_ema

    def train_dataloader(self):
        args = self.hparams
        transform = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True),
            ]
        )

        dataset = MultiResolutionDataset(args.path, transform, args.size)
        dataloader = data.DataLoader(
            dataset,
            shuffle=True,
            batch_size=args.batch_size,
            drop_last=True,
            num_workers=args.num_workers,
        )
        return dataloader

    def forward(self, z):
        return self.generator(z)

    def discriminator_loss(self, real_img):
        args = self.hparams
        requires_grad(self.generator, False)
        requires_grad(self.discriminator, True)

        noise = mixing_noise(args.batch_size, args.latent, args.mixing, self.device)
        fake_img, _ = self.generator(noise)
        real_img_aug = real_img

        if args.augment:
            real_img_aug, _ = augment(real_img, self.ada_aug_p)
            fake_img, _ = augment(fake_img, self.ada_aug_p)

        fake_pred = self.discriminator(fake_img)
        real_pred = self.discriminator(real_img_aug)
        d_loss = d_logistic_loss(real_pred, fake_pred)

        real_score = real_pred.mean()
        fake_score = fake_pred.mean()

        if args.augment and args.augment_p == 0:
            self.ada_augment += self.ada_augment.new_tensor([torch.sign(real_pred).sum().item(), real_pred.shape[0]])
            #ada_augment = reduce_sum(ada_augment)
            if self.ada_augment[1] > 255:
                pred_signs, n_pred = self.ada_augment.tolist()
                r_t_stat = pred_signs / n_pred
                sign = 1 if r_t_stat > args.ada_target else -1
                self.ada_aug_p += sign * self.ada_aug_step * n_pred
                self.ada_aug_p = min(1, max(0, self.ada_aug_p))
                self.ada_augment.mul_(0)

        return d_loss, real_score, fake_score

    def discriminator_regularization_loss(self, real_img):
        args = self.hparams
        real_img.requires_grad = True
        real_pred = self.discriminator(real_img)
        r1_loss = d_r1_loss(real_pred, real_img)
        d_reg_loss = (args.r1 / 2 * r1_loss * args.d_reg_every + 0 * real_pred[0]).mean()
        return d_reg_loss, r1_loss

    def generator_loss(self):
        args = self.hparams
        requires_grad(self.generator, True)
        requires_grad(self.discriminator, False)

        noise = mixing_noise(args.batch_size, args.latent, args.mixing, self.device)
        fake_img, _ = self.generator(noise)

        if args.augment:
            fake_img, _ = augment(fake_img, self.ada_aug_p)

        fake_pred = self.discriminator(fake_img)
        g_loss = g_nonsaturating_loss(fake_pred)
        return g_loss

    def generator_regularization_loss(self):
        args = self.hparams
        path_batch_size = max(1, args.batch_size // args.path_batch_shrink)
        noise = mixing_noise(path_batch_size, args.latent, args.mixing, self.device)
        fake_img, latents = self.generator(noise, return_latents=True)

        path_loss, self.mean_path_length, path_lengths = g_path_regularize(
            fake_img, latents, self.mean_path_length
        )
        # generator.zero_grad()
        weighted_path_loss = args.path_regularize * args.g_reg_every * path_loss

        if args.path_batch_shrink:
            weighted_path_loss += 0 * fake_img[0, 0, 0, 0]

        return weighted_path_loss, path_loss, path_lengths

    def discriminator_step(self, real_img, regularize=False):
        d_loss, real_score, fake_score = self.discriminator_loss(real_img)

        tqdm_dict = {'d_loss': d_loss}
        log_dict = {'d_loss': d_loss, 'real_score': real_score, 'fake_score': fake_score}

        if regularize:
            d_reg_loss, r1_loss = self.discriminator_regularization_loss(real_img)
            d_loss += d_reg_loss
            log_dict.update({'r1': r1_loss})

        output = {
            'loss': d_loss,
            'progress_bar': tqdm_dict,
            'log': log_dict,
        }
        return output

    def generator_step(self, regularize=False):
        g_loss = self.generator_loss()
        tqdm_dict = {'g_loss': g_loss}
        log_dict = {'g_loss': g_loss}

        if regularize:
            weighted_path_loss, path_loss, path_lengths = self.generator_regularization_loss()
            # weighted_path_loss.backward()
            # g_optim.step()
            g_loss += weighted_path_loss
            log_dict["path"] = path_loss
            log_dict["path_length"] = path_lengths.mean()

        output = {
            'loss': g_loss,
            'progress_bar': tqdm_dict,
            'log': log_dict,
        }
        return output

    def training_step(self, batch, batch_idx, optimizer_idx):
        args = self.hparams
        d_regularize = (batch_idx % self.hparams.d_reg_every == 0)
        g_regularize = (batch_idx % args.g_reg_every == 0)
        real_img = batch
        result = None
        if optimizer_idx == 0:
            result = self.discriminator_step(real_img, regularize=d_regularize)

        if optimizer_idx == 1:
            result = self.generator_step(regularize=g_regularize)
            accumulate(self.g_ema, self.generator, self.accum)

        if batch_idx % args.img_log_frequency == 0:
            with torch.no_grad():
                self.g_ema.eval()
                sample, _ = self.g_ema([self.sample_z])
                grid = utils.make_grid(
                    sample,
                    nrow=int(args.n_sample ** 0.5),
                    normalize=True,
                    range=(-1, 1),
                )
                self.logger.experiment.log({
                    'examples': [wandb.Image(grid.cpu())],
                }, commit=False)

        self.print(result['loss'])

        return result


def main(args):
    seed_everything(234)
    model = StyleGAN2(args)
    logger = WandbLogger(project='stylegan2')
    trainer = Trainer.from_argparse_args(
        args,
        logger=logger,
        max_steps=800000,
    )
    trainer.fit(model)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=str)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--n_sample", type=int, default=64)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--r1", type=float, default=10)
    parser.add_argument("--path_regularize", type=float, default=2)
    parser.add_argument("--path_batch_shrink", type=int, default=2)
    parser.add_argument("--d_reg_every", type=int, default=16)
    parser.add_argument("--g_reg_every", type=int, default=4)
    parser.add_argument("--mixing", type=float, default=0.9)
    parser.add_argument("--lr", type=float, default=0.002)
    parser.add_argument("--channel_multiplier", type=int, default=2)
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--augment_p", type=float, default=0)
    parser.add_argument("--ada_target", type=float, default=0.6)
    parser.add_argument("--ada_length", type=int, default=500 * 1000)
    parser.add_argument("--img_log_frequency", type=int, default=100)
    parser.add_argument("--num_workers", type=int, default=8)
    parser = Trainer.add_argparse_args(parser)
    params = parser.parse_args()
    main(params)

