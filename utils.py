
import shutil
import math
import os
import torch
import torchvision
import torchvision.transforms as transforms


def data_augmentation(config, is_train=True):
    aug = []
    if is_train:
        # random crop
        if config.augmentation.is_random_crop:
            aug.append(transforms.RandomCrop(config.input_size, padding=4))
        # horizontal filp
        if config.augmentation.is_random_horizontal_filp:
            aug.append(transforms.RandomHorizontalFlip())

    aug.append(transforms.ToTensor())
    # normalize  [- mean / std]
    if config.augmentation.is_normalize:
        if config.dataset == 'cifar10':
            aug.append(transforms.Normalize(
                (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)))
        else:
            aug.append(transforms.Normalize(
                (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)))
    return aug


def save_checkpoint(state, is_best, filename):
    torch.save(state, filename+'.pth.tar')
    if is_best:
        shutil.copyfile(filename+'.pth.tar', filename+'_best.pth.tar')


def load_checkpoint(path, model, optimizer=None):
    if os.path.isfile(path):
        print("=== loading checkpoint '{}' ===".format(path))

        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint['state_dict'], strict=False)

        if optimizer != None:
            best_prec = checkpoint['best_prec']
            last_epoch = checkpoint['last_epoch']
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=== done. also loaded optimizer from checkpoint '{}' (epoch {}) ===".format(
                path, last_epoch + 1))
            return best_prec, last_epoch


def get_data_loader(transform_train, transform_test, config):
    assert config.dataset == 'cifar10' or config.dataset == 'cifar100'
    if config.dataset == "cifar10":
        trainset = torchvision.datasets.CIFAR10(
            root=config.data_path, train=True, download=True, transform=transform_train)

        testset = torchvision.datasets.CIFAR10(
            root=config.data_path, train=False, download=True, transform=transform_test)
    else:
        trainset = torchvision.datasets.CIFAR100(
            root=config.data_path, train=True, download=True, transform=transform_train)

        testset = torchvision.datasets.CIFAR100(
            root=config.data_path, train=False, download=True, transform=transform_test)

    train_loader = torch.utils.data.DataLoader(
        trainset, batch_size=config.batch_size, shuffle=True, num_workers=config.workers)

    test_loader = torch.utils.data.DataLoader(
        testset, batch_size=config.test_batch, shuffle=False, num_workers=config.workers)
    return train_loader, test_loader


def get_current_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def adjust_learning_rate(optimizer, epoch, config):
    lr = get_current_lr(optimizer)
    if config.lr_scheduler.type == 'STEP':
        if epoch in config.lr_scheduler.lr_epochs:
            lr *= config.lr_scheduler.lr_mults
    elif config.lr_scheduler.type == 'COSINE':
        ratio = epoch / config.epochs
        lr = config.lr_scheduler.min_lr + \
            (config.lr_scheduler.base_lr - config.lr_scheduler.min_lr) * \
            (1.0 + math.cos(math.pi * ratio)) / 2.0
    elif config.lr_scheduler.type == 'HTD':
        ratio = epoch / config.epochs
        lr = config.lr_scheduler.min_lr + \
            (config.lr_scheduler.base_lr - config.lr_scheduler.min_lr) * \
            (1.0 - math.tanh(config.lr_scheduler.lower_bound +
                             (config.lr_scheduler.upper_bound - config.lr_scheduler.lower_bound) * ratio)) / 2.0
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
