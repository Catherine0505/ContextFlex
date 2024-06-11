import torch
import torch.nn.functional as nnf
import torch.nn as nn
import torch.nn.functional as F

import einops
import math

def check_dims(x, ndim): 
    assert x.ndim == ndim, \
        f"Expect tensor to have {ndim} dimensions but got shape: {x.shape}."