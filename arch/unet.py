""" Full assembly of the parts to form the complete network """

import math
from DRDiff.arch.module import *
import torch

class PositionalEncoding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        count = self.dim // 2
        step = torch.arange(count, dtype=time.dtype,
                            device=time.device) / count
        encoding = time.unsqueeze(1) * torch.exp(-math.log(1e4) * step.unsqueeze(0))
        encoding = torch.cat(
            [torch.sin(encoding), torch.cos(encoding)], dim=-1)
        # encoding shape: [1, 1, dim] (dim=32)
        return encoding

class UNet(nn.Module):
    def __init__(self, n_channels, scale_factor, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.scale_factor = scale_factor
        self.bilinear = bilinear

        self.inc = (DoubleConv(10, 16))
        self.down1 = (Down(16, 32))
        self.down2 = (Down(32, 64))
        self.down3 = (Down(64, 128))
        factor = 2 if bilinear else 1
        self.down4 = (Down(128, 256 // factor))
        self.up1 = (Up(256, 128 // factor, bilinear))
        self.up2 = (Up(128, 64 // factor, bilinear))
        self.up3 = (Up(64, 32 // factor, bilinear))
        self.up4 = (Up(32, 16, bilinear))
        self.outc = (OutConv(16, n_channels))

        self.t_embedding = nn.Sequential(
            PositionalEncoding(4),
            nn.Linear(4, 8),
            Swish(),
            nn.Linear(8, 8)
        )

        self.initialize_weights()

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.normal_(m.weight.data, 0, 0.01)
                # torch.nn.init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, torch.nn.ConvTranspose2d):
                torch.nn.init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, torch.nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, torch.nn.Linear):
                torch.nn.init.normal_(m.weight.data, 0, 0.01)
                m.bias.data.zero_()

    def forward(self, xt, time, lr = None):
        t = self.t_embedding(time).unsqueeze(-1).unsqueeze(-1)
        n, _, h, w = xt.shape
        t = t.expand(n, -1, h, w)
        if lr is not None:
            if xt.shape != lr.shape:
                lr = F.interpolate(lr, scale_factor=self.scale_factor, mode='bicubic', align_corners=False)

        x = torch.cat([xt, lr, t], dim=1)
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return lr + logits

    def use_checkpointing(self):
        self.inc = torch.utils.checkpoint(self.inc)
        self.down1 = torch.utils.checkpoint(self.down1)
        self.down2 = torch.utils.checkpoint(self.down2)
        self.down3 = torch.utils.checkpoint(self.down3)
        self.down4 = torch.utils.checkpoint(self.down4)
        self.up1 = torch.utils.checkpoint(self.up1)
        self.up2 = torch.utils.checkpoint(self.up2)
        self.up3 = torch.utils.checkpoint(self.up3)
        self.up4 = torch.utils.checkpoint(self.up4)
        self.outc = torch.utils.checkpoint(self.outc)


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

# if __name__ == '__main__':
#     a = torch.randn(1, 1, 1320, 1680).cuda()
#     lr = torch.randn(1, 1, 1320, 1680).cuda()
#     t = torch.randn(1).cuda()
#     model = UNet(1, 12).cuda()
#     total_params = sum(p.numel() for p in model.parameters())
#     print("total #param: ", total_params)