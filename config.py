import argparse
import platform
from utils import process_config

import torch


def get_train_config(ViT=False):
    # Training settings
    # Hardware specifications
    parser = argparse.ArgumentParser(description='PyTorch BVRotNet')
    parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
    parser.add_argument('--gpus', default=8, type=float, help='number of gpu')
    parser.add_argument('--id', default=0, type=float, help='id of gpu')
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
    parser.add_argument("--dataset", type=str, default='ImageNet', help="dataset for fine-tunning/evaluation")
    parser.add_argument("--image-size", type=int, default=128, help="input image size", choices=[224, 384])

    # Model specifications
    if ViT:
        parser.add_argument('--model_type', type=str, default='ViT')
        parser.add_argument('--cnn', type=bool, default=False)
    else:
        parser.add_argument('--model_type', type=str, default='MobileNet')
        parser.add_argument('--cnn', type=bool, default=True)
    parser.add_argument("--num-classes", type=int, default=101, help="number of classes in dataset")
    parser.add_argument('--pretrained_model', default='imagenet21k+imagenet2012_ViT-B_16-224.pth',
                        help='ViT pretrained base model')
    parser.add_argument('--pretrained', type=bool, default=False)
    parser.add_argument('--prefix', default='tq', help='Location to save checkpoint models')
    parser.add_argument('--save_folder', default='./weights/',
                        help='Location to save checkpoint models')

    # Training specifications
    parser.add_argument('--batchSize', type=int, default=32, help='training batch size')
    parser.add_argument('--nEpochs', type=int, default=120, help='number of epochs to train for')
    parser.add_argument('--snapshots', type=int, default=30, help='Snapshots')
    parser.add_argument("--train-steps", type=int, default=10000000000, help="number of training/fine-tunning steps")

    # Optimization specifications
    if ViT:
        parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
    else:
        parser.add_argument('--lr', type=float, default=1.0e-4, help='learning rate')
    parser.add_argument("--wd", type=float, default=1e-4, help='weight decay')
    parser.add_argument("--warmup-steps", type=int, default=500, help='learning rate warm up steps')

    # ViT config
    parser.add_argument("--model-arch", type=str, default="b16", help='model setting to use',
                        choices=['b16', 'b32', 'l16', 'l32', 'h14'])

    parser.add_argument("--exp-name", type=str, default="train", help="experiment name")

    config = parser.parse_args()

    # model config
    config = eval("get_{}_config".format(config.model_arch))(config)
    process_config(config)
    print_config(config)
    return config


def get_b16_config(config):
    """ ViT-B/16 configuration """
    config.patch_size = 16
    config.emb_dim = 64
    config.mlp_dim = 256
    config.num_heads = 4
    config.num_layers = 4
    config.attn_dropout_rate = 0.0
    config.dropout_rate = 0.1
    return config


def get_b32_config(config):
    """ ViT-B/32 configuration """
    config = get_b16_config(config)
    config.patch_size = 32
    return config


def get_l16_config(config):
    """ ViT-L/16 configuration """
    config.patch_size = 16
    config.emb_dim = 1024
    config.mlp_dim = 4096
    config.num_heads = 16
    config.num_layers = 24
    config.attn_dropout_rate = 0.0
    config.dropout_rate = 0.1
    return config


def get_l32_config(config):
    """ Vit-L/32 configuration """
    config = get_l16_config(config)
    config.patch_size = 32
    return config


def get_h14_config(config):
    """  ViT-H/14 configuration """
    config.patch_size = 14
    config.emb_dim = 1280
    config.mlp_dim = 5120
    config.num_heads = 16
    config.num_layers = 32
    config.attn_dropout_rate = 0.0
    config.dropout_rate = 0.1
    return config


def print_config(config):
    message = ''
    message += '----------------- Config ---------------\n'
    for k, v in sorted(vars(config).items()):
        comment = ''
        message += '{:>35}: {:<30}{}\n'.format(str(k), str(v), comment)
    message += '----------------- End -------------------'
    print(message)
