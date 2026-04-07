
import numpy as np
import torch

def calc_ssim(x, y):
    x, y = x.cpu().squeeze().numpy(), y.cpu().squeeze().numpy()
    mu_x, mu_y = np.mean(x), np.mean(y)
    sigma_x, sigma_y = np.std(x), np.std(y)
    # sigma_xy = (1./((x.flatten()).size -1)) * np.sum((x.flatten()-mu_x)*(y.flatten()-mu_y))
    sigma_xy = np.cov(x, y, bias=True)[0, 1]
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    # c3 = c2 / 2.
    ssim = ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / \
           ((mu_x ** 2 + mu_y ** 2 + c1) * (sigma_x ** 2 + sigma_y ** 2 + c2))
    return ssim

def calc_psnr(img1, img2):
    return 10. * torch.log10(1. / torch.mean((img1 - img2) ** 2))

def calc_rmse(img1, img2):
    return torch.sqrt(torch.mean((img1 - img2)**2))

def calc_mae(img1, img2):
    return torch.mean(torch.abs(img1 - img2))

def calc_mbe(img1, img2):
    return torch.mean(img1 - img2)

def calc_r2(img1, img2):
    ss_res = torch.sum((img2 - img1) ** 2)
    ss_tot = torch.sum((img2 - torch.mean(img2)) ** 2)
    if ss_tot == 0:
        return torch.tensor(1.0 if ss_res == 0 else float('nan'))
    r2 = 1 - ss_res / ss_tot
    return r2