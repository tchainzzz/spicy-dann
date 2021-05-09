import numpy as np
import torch

def mixup_data(x, lam, use_cuda=True):
    batch_size = x.size(0)
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    return mixed_x, index, lam


def mixup_criterion(criterion, pred, y, permutation, lam):
    y_mix = y[permutation]
    return lam * criterion(pred, y) + (1 - lam) * criterion(pred, y_mix)
