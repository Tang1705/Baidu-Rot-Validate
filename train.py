import os
import time
import socket

import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader

import utils
from Data import get_training_set
from BVRotNet import BVRotNet

from config import get_train_config

opt = get_train_config()
gpus_list = range(opt.gpus)
hostname = str(socket.gethostname())


def train(epoch):
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
        if not opt.cnn:
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


def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)


def checkpoint(epoch):
    model_out_path = opt.checkpoint_dir + opt.model_type + "_epoch_{}.pth".format(
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

model = BVRotNet(opt)

print('---------- Networks architecture -------------')
print_network(model)
print('----------------------------------------------')

if opt.pretrained:
    model_name = os.path.join(opt.save_folder + opt.pretrained_model)
    if os.path.exists(model_name):
        pretrained_dict = torch.load(model_name, map_location=lambda storage, loc: storage)["state_dict"]
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if
                           (k in model_dict and 'classifier' not in k)}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

        print(
            '************************************Pre-trained model is loaded.************************************')
    else:
        print(
            '************************************Pre-trained model isn\'t loaded.************************************')

criterion = nn.CrossEntropyLoss()

# create optimizers and learning rate scheduler
if opt.cnn:
    optimizer = model.get_optimizer()
else:
    optimizer, lr_scheduler = model.get_optimizer()

if opt.gpu_mode:
    model = model.cuda(gpus_list[opt.id])
    criterion = criterion.cuda(gpus_list[opt.id])

for epoch in range(1, opt.nEpochs + 1):
    train(epoch)

    if (epoch + 1) % 30 == 0:
        for param_group in optimizer.param_groups:
            param_group['lr'] /= 10.0
        print('Learning rate decay: lr={}'.format(optimizer.param_groups[0]['lr']))

    if (epoch + 1) % (opt.snapshots) == 0 or (epoch + 1) == opt.nEpochs:
        checkpoint(epoch)