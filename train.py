import argparse
import yaml
import time
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn

from easydict import EasyDict
from models import *
from utils import *

parser = argparse.ArgumentParser(description='PyTorch CIFAR Dataset Training')
parser.add_argument('--work-path', required=True, type=str)
parser.add_argument('--resume', action='store_true',
                    help='resume from checkpoint')


def train(train_loader, net, criterion, optimizer, epoch, device):

    start = time.time()
    net.train()

    train_loss = 0
    correct = 0
    total = 0
    print(" === Epoch: [{}/{}] === ".format(epoch + 1, config.epochs))

    for batch_index, (inputs, targets) in enumerate(train_loader):
        # move tensor to GPU
        inputs, targets = inputs.to(device), targets.to(device)
        # zero the gradient buffers
        optimizer.zero_grad()
        # forward
        outputs = net(inputs)
        # cal the loss
        loss = criterion(outputs, targets)
        # backward
        loss.backward()
        # update weight
        optimizer.step()

        # count the loss and acc
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        if (batch_index + 1) % 100 == 0:
            print("   == step: [{:3}/{}], train loss: {:.3f} | train acc: {:6.3f}% | lr: {:.6f}".format(
                batch_index + 1, len(train_loader),
                train_loss/(batch_index+1), 100.0*correct/total, get_current_lr(optimizer)))

    print("   == step: [{:3}/{}], train loss: {:.3f} | train acc: {:6.3f}% | lr: {:.6f}".format(
        batch_index + 1, len(train_loader),
        train_loss/(batch_index+1), 100.0*correct/total, get_current_lr(optimizer)))

    end = time.time()
    print("   == cost time: {:.4f}s".format(end - start))
    return train_loss / (batch_index+1), correct / total


def test(test_loader, net, criterion, optimizer, epoch, device):
    global best_prec

    net.eval()

    test_loss = 0
    correct = 0
    total = 0

    print(" === Validate ===".format(epoch + 1, config.epochs))

    with torch.no_grad():
        for batch_index, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    print("   == test loss: {:.3f} | test acc: {:6.3f}%".format(
        test_loss/(batch_index+1), 100.0*correct/total))

    # Save checkpoint.
    acc = 100.*correct/total
    state = {
        'state_dict': net.state_dict(),
        'best_prec': best_prec,
        'last_epoch': epoch,
        'optimizer': optimizer.state_dict(),
    }
    is_best = acc > best_prec
    save_checkpoint(state, is_best, args.work_path + '/' + config.ckpt_name)
    if is_best:
        best_prec = acc


def main():
    global args, config, last_epoch, best_prec
    args = parser.parse_args()

    # read config from yaml file
    with open(args.work_path + '/config.yaml') as f:
        config = yaml.load(f)
    # convert to dict
    config = EasyDict(config)

    # define netowrk
    net = get_model(config)

    # print args and network architecture
    print(args)
    print(net)

    # CPU or GPU
    device = 'cuda' if config.use_gpu else 'cpu'
    # data parallel for multiple-GPU
    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True

    net.to(device)

    # define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), config.lr_scheduler.base_lr,
                                momentum=config.optimize.momentum,
                                weight_decay=config.optimize.weight_decay, nesterov=config.optimize.nesterov)

    # resume from a checkpoint
    last_epoch = -1
    best_prec = 0
    if args.work_path:
        if args.resume:
            best_prec, last_epoch = load_checkpoint(
                args.work_path, net, optimizer=optimizer)
        else:
            load_checkpoint(args.work_path, net)

    # load training data, do data augmentation and get data loader
    transform_train = transforms.Compose(
        data_augmentation(config))

    transform_test = transforms.Compose(
        data_augmentation(config, is_train=False))

    train_loader, test_loader = get_data_loader(
        transform_train, transform_test, config)

    print("\n=======  Training  =======\n")
    for epoch in range(last_epoch + 1, config.epochs):
        adjust_learning_rate(optimizer, epoch, config)
        train(train_loader, net, criterion, optimizer, epoch, device)
        if epoch == 0 or (epoch + 1) % config.eval_freq == 0 or epoch == config.epochs - 1:
            test(test_loader, net, criterion, optimizer, epoch, device)


if __name__ == "__main__":
    main()
