
import torch

def image_compare_loss(x, y):
    loss = torch.norm(x - y, p=2)
    return loss