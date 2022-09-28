
def compute_acc(pred, label, reduction='mean'):
  result = (pred == label).float()
  if reduction == 'none':
    return result.detach()
  elif reduction == 'mean':
    return result.mean().item()


class AverageMeter(object):
  def __init__(self):
    self.reset()

  def reset(self):
    self.val = 0.
    self.avg = 0.
    self.sum = 0.
    self.count = 0.

  def update(self, val, n=1):
    self.val = val
    self.sum += val * n
    self.count += n
    self.avg = self.sum / self.count

  def item(self):
    return self.avg