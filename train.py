import os
import torch
import argparse
import platform
import socket
import time

import torchvision.models
from torch import nn
from torch import optim

from config import get_train_config
from Data import get_training_set
from torch.autograd import Variable
from torch.utils.data import DataLoader
from VisionTransformer import VisionTransformer

# Training settings
# Hardware specifications
parser = argparse.ArgumentParser(description='PyTorch ViT')
parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
parser.add_argument('--gpus', default=8, type=float, help='number of gpu')
parser.add_argument('--id', default=4, type=float, help='id of gpu')
if torch.cuda.is_available():
    parser.add_argument('--gpu_mode', type=bool, default=False)
else:
    parser.add_argument('--gpu_mode', type=bool, default=False)

plat_tuple = platform.architecture()
system = platform.system()
plat_version = platform.platform()
if system == 'Linux':
    parser.add_argument('--threads', type=int, default=10, help='number of threads for data loader to use')
else:
    parser.add_argument('--threads', type=int, default=0, help='number of threads for data loader to use')

# Data specifications
parser.add_argument('--data_dir', type=str, default='./data')

# Model specifications
# parser.add_argument('--model_type', type=str, default='ResNet50')
parser.add_argument('--model_type', type=str, default='ViT')
parser.add_argument('--pretrained', default='imagenet21k+imagenet2012_ViT-B_16-224.pth',
                    help='ViT pretrained base model')
parser.add_argument('--pretrained_vit', type=bool, default=False)
parser.add_argument('--prefix', default='tq', help='Location to save checkpoint models')
parser.add_argument('--save_folder', default='./weights/',
                    help='Location to save checkpoint models')

# Training specifications
parser.add_argument('--batchSize', type=int, default=32, help='training batch size')
parser.add_argument('--nEpochs', type=int, default=120, help='number of epochs to train for')
parser.add_argument('--snapshots', type=int, default=30, help='Snapshots')

# Optimization specifications
parser.add_argument('--lr', type=float, default=1.0e-4, help='learning rate')

opt = parser.parse_args()
gpus_list = range(opt.gpus)
hostname = str(socket.gethostname())
print(opt)


def train(epoch):
    global min
    epoch_loss = 0
    model.train()
    t_start = time.time()
    for iteration, batch in enumerate(training_data_loader, 1):
        input, target = Variable(batch[0]), batch[1]

        if opt.gpu_mode:
            input = input.cuda(gpus_list[opt.id])
            target = target.cuda(gpus_list[opt.id])

        t0 = time.time()
        optimizer.zero_grad()
        pred = model(input)
        loss = criterion(pred, target)

        epoch_loss += loss

        loss.backward()
        optimizer.step()
        if opt.model_type == "ViT":
            lr_scheduler.step()

        t1 = time.time()

        print(
            "===> Epoch[{}]({}/{}): Loss: {:.4f} || Timer: {:.4f} sec.".format(
                epoch, iteration, len(training_data_loader), loss.item(), t1 - t0))
    t_end = time.time()
    print(
        "===> Epoch {} Complete: Avg. Loss: {:.4f} || Total Time: {:.4f} min.".format(epoch,
                                                                                      epoch_loss / len(
                                                                                          training_data_loader),

                                                                                      (
                                                                                              t_end - t_start) / 60),
        time.ctime())

    if epoch_loss / len(training_data_loader) < min:
        min = epoch_loss / len(training_data_loader)
        checkpoint(epoch)


def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)


def checkpoint(epoch):
    model_out_path = opt.save_folder + opt.model_type + "_epoch_{}.pth".format(
        epoch)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
    }, model_out_path)
    print("Checkpoint saved to {}".format(model_out_path))


torch.manual_seed(opt.seed)
if opt.gpu_mode:
    torch.cuda.manual_seed(opt.seed)

print('===> Loading datasets')
train_set = get_training_set(opt.data_dir)
training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=True)

print('===> Building model ', opt.model_type)

config = get_train_config()
if opt.model_type == "ViT":
    model = VisionTransformer(
        image_size=(config.image_size, config.image_size),
        patch_size=(config.patch_size, config.patch_size),
        emb_dim=config.emb_dim,
        mlp_dim=config.mlp_dim,
        num_heads=config.num_heads,
        num_layers=config.num_layers,
        num_classes=config.num_classes,
        attn_dropout_rate=config.attn_dropout_rate,
        dropout_rate=config.dropout_rate)
else:
    model = torchvision.models.mobilenet_v3_small(pretrained=True)
    model.classifier = nn.Sequential(
        nn.Linear(in_features=576, out_features=1024, bias=True),
        nn.Hardswish(),
        nn.Dropout(p=0.2, inplace=True),
        nn.Linear(in_features=1024, out_features=101, bias=True))
    # model = torchvision.models.resnet50(pretrained=True)
    # fc_features = model.fc.in_features
    # model.fc = nn.Linear(fc_features, 101)

print('---------- Networks architecture -------------')
print_network(model)
print('----------------------------------------------')

if opt.pretrained_vit:
    model_name = os.path.join(opt.save_folder + opt.pretrained)
    if os.path.exists(model_name):
        pretrained_dict = torch.load(model_name, map_location=lambda storage, loc: storage)["state_dict"]
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if
                           (k in model_dict and 'classifier' not in k)}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

        print(
            '************************************Pre-trained ViT model is loaded.************************************')
    else:
        print(
            '************************************Pre-trained ViT model isn\'t loaded.************************************')

criterion = nn.CrossEntropyLoss()

# create optimizers and learning rate scheduler
if opt.model_type == "ViT":
    optimizer = torch.optim.SGD(
        params=model.parameters(),
        lr=config.lr,
        weight_decay=config.wd,
        momentum=0.9)
    lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer=optimizer,
        max_lr=config.lr,
        pct_start=config.warmup_steps / config.train_steps,
        total_steps=config.train_steps)
else:
    optimizer = optim.Adam(model.parameters(), lr=config.lr, betas=(0.9, 0.999), weight_decay=config.wd, eps=1e-8)

if opt.gpu_mode:
    model = model.cuda(gpus_list[opt.id])
    criterion = criterion.cuda(gpus_list[opt.id])

min = 10

for epoch in range(1, opt.nEpochs + 1):
    train(epoch)

    if (epoch + 1) % 30 == 0:
        for param_group in optimizer.param_groups:
            param_group['lr'] /= 10.0
        print('Learning rate decay: lr={}'.format(optimizer.param_groups[0]['lr']))

    # if (epoch + 1) % (opt.snapshots) == 0 or (epoch + 1) == opt.nEpochs:
    #     checkpoint(epoch)