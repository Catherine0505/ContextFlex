import torch
import torch.nn as nn
import torch.nn.functional as F
from . import utils

import einops

import warnings

def harddice(y_pred, y_true, dim, eps=1e-3): 

    num_classes = y_pred.shape[dim]
    y_pred_hard = F.one_hot(y_pred.argmax(dim=dim), num_classes=num_classes)
    y_true_hard = F.one_hot(y_true.argmax(dim=dim), num_classes=num_classes)

    sum_dims = tuple(range(dim, y_pred_hard.ndim - 1))

    # compute dice
    num = 2 * (y_pred_hard * y_true_hard).sum(dim=sum_dims) + eps
    denom = (y_pred_hard ** 2).sum(dim=sum_dims) + \
        (y_true_hard ** 2).sum(dim=sum_dims) + eps
    score = num / denom
    return score