import torch.optim
from torch.optim.lr_scheduler import MultiStepLR, CosineAnnealingLR

from core.workspace import Registers


@Registers.optimizer.register
def SGD(params=None, lr=1e-3, momentum=0.9, weight_decay=0.):
    return torch.optim.SGD(params, lr, momentum=0.9, weight_decay=weight_decay)


@Registers.optimizer.register
def RMSprop(params=None, lr=1e-3, momentum=0.9, weight_decay=0.):
    return torch.optim.RMSprop(params, lr, weight_decay=weight_decay)


@Registers.optimizer.register
def Adam(params, lr, weight_decay=0.):
    return torch.optim.Adam(params, lr, weight_decay=weight_decay)
