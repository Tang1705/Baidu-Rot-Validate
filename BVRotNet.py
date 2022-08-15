import torch
import torch.nn as nn
import torchvision
from torch import optim

from VisionTransformer import VisionTransformer


class BVRotNet(nn.Module):
    def __init__(self, opt):
        super(BVRotNet, self).__init__()

        self.opt = opt

        if opt.model_type == "ViT":
            self.model = VisionTransformer(
                image_size=(self.opt.image_size, self.opt.image_size),
                patch_size=(self.opt.patch_size, self.opt.patch_size),
                emb_dim=self.opt.emb_dim,
                mlp_dim=self.opt.mlp_dim,
                num_heads=self.opt.num_heads,
                num_layers=self.opt.num_layers,
                num_classes=self.opt.num_classes,
                attn_dropout_rate=self.opt.attn_dropout_rate,
                dropout_rate=self.opt.dropout_rate)
        else:
            self.model = torchvision.models.mobilenet_v3_small(pretrained=True)
            self.model.classifier = nn.Sequential(
                nn.Linear(in_features=576, out_features=1024, bias=True),
                nn.Hardswish(),
                nn.Dropout(p=0.2, inplace=True),
                nn.Linear(in_features=1024, out_features=101, bias=True))

    def forward(self, x):
        return self.model(x)

    def get_optimizer(self):
        if self.opt.model_type == "ViT":
            optimizer = torch.optim.SGD(
                params=self.model.parameters(),
                lr=self.opt.lr,
                weight_decay=self.opt.wd,
                momentum=0.9)
            lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer=optimizer,
                max_lr=self.opt.lr,
                pct_start=self.opt.warmup_steps / self.opt.train_steps,
                total_steps=self.opt.train_steps)
            return optimizer, lr_scheduler
        else:
            optimizer = optim.Adam(self.model.parameters(), lr=self.opt.lr, betas=(0.9, 0.999),
                                   weight_decay=self.opt.wd,
                                   eps=1e-8)

            return optimizer
