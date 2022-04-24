import torch.nn.functional as F
import torch
import time
from pretrain_dataloader import *

def update_target_params(online_params, target_params, tau):

    #update the backbone first
    for op, mp in zip(online_params, target_params):
        mp.data = tau * mp.data + (1 - tau) * op.data

def set_params_equal(online_params, target_params):

    #update the backbone first
    for op, mp in zip(online_params, target_params):
        mp.data = op.data
    
def byol_loss_func(p: torch.Tensor, z: torch.Tensor, simplified: bool = True) -> torch.Tensor:
    return 2 - 2 * F.cosine_similarity(p, z.detach(), dim=-1).mean()


def prepare_cifar_train_loader(args):
    transform = [
        prepare_transform(args.dataset, **kwargs) for kwargs in args.transform_kwargs
    ]

    transform = prepare_n_crop_transform(transform, num_crops_per_aug=args.num_crops_per_aug)

    train_dataset = prepare_datasets(
        args.dataset,
        transform,
        no_labels=False,
    )
    train_loader = prepare_dataloader(
        train_dataset, batch_size=args.batch_size, num_workers=args.num_workers
    )
    return train_loader


class args:
    def __init__(self):
        return
        

def return_default_args(args = args()):
    args.dataset = 'cifar100'
    args.transform_kwargs=[{'brightness': 0.4, 'contrast': 0.4, 'saturation': 0.2, 'hue': 0.1, 'color_jitter_prob': 0.8, 'gray_scale_prob': 0.2, 'horizontal_flip_prob': 0.5, 'gaussian_prob': 1.0, 'solarization_prob': 0.0, 'crop_size': 32, 'min_scale': 0.08, 'max_scale': 1.0}, {'brightness': 0.4, 'contrast': 0.4, 'saturation': 0.2, 'hue': 0.1, 'color_jitter_prob': 0.8, 'gray_scale_prob': 0.2, 'horizontal_flip_prob': 0.5, 'gaussian_prob': 0.1, 'solarization_prob': 0.2, 'crop_size': 32, 'min_scale': 0.08, 'max_scale': 1.0}]
    # asymmetric augmentations
    args.num_crops_per_aug = [1, 1]
    args.batch_size = 256
    args.num_workers = 4
    
    args.optimizer = 'LARS'
    args.lr = .3
    args.weight_decay = 1.5e-6
    args.momentum = .9
    args.classifier_lr = .1
    args.classifier_weight_decay = 0
    args.epochs = 200
    args.warmup_epochs = 10
    args.steps = int((50000/args.batch_size) * args.epochs)
    args.warmup_steps = int((50000/args.batch_size) * args.warmup_epochs)
    return args
    