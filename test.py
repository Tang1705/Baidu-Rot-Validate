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
parser.add_argument('--data_dir', type=str, default='./val')

# Model specifications
parser.add_argument('--model_type', type=str, default='ViT')
# parser.add_argument('--model_type', type=str, default='ViT')
parser.add_argument('--pretrained', default='ViT_epoch_67.pth',
                    help='ViT pretrained base model')

# Testing specifications
parser.add_argument('--batchSize', type=int, default=1, help='training batch size')

opt = parser.parse_args()
gpus_list = range(opt.gpus)
hostname = str(socket.gethostname())
print(opt)


def test():
    model.eval()
    for iteration, batch in enumerate(testing_data_loader, 1):
        input, target = Variable(batch[0]), batch[1]

        if opt.gpu_mode:
            input = input.cuda(gpus_list[opt.id])
            target = target.cuda(gpus_list[opt.id])

        t0 = time.time()
        pred = model(input)
        acc = torch.argmax(pred) - target

        t1 = time.time()

        print(
            "===> ({}/{}): Acc: {:.4f} || Timer: {:.4f} sec.".format(iteration, len(testing_data_loader), acc.item(),
                                                                     t1 - t0))

print('===> Loading datasets')
test_set = get_training_set(opt.data_dir)
testing_data_loader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=False)

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
    model = torchvision.models.mobilenet_v3_small()
    model.classifier = nn.Sequential(
        nn.Linear(in_features=576, out_features=1024, bias=True),
        nn.Hardswish(),
        nn.Dropout(p=0.2, inplace=True),
        nn.Linear(in_features=1024, out_features=101, bias=True))
    model.load_state_dict(torch.load("./weights/ResNet50_epoch_29.pth")["model_state_dict"])

if opt.gpu_mode:
    model = model.cuda(gpus_list[opt.id])

test()
