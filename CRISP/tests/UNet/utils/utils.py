import torch
from torchvision import transforms
import torch.nn.functional as F

def get_transform_inv_normalize():
    """Get Imagenet normalization transform function

    Returns:
        inverse normalization function
    """
    return transforms.Normalize(
        mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.255],
        std=[1 / 0.229, 1 / 0.224, 1 / 0.255]
    )


def to_one_hot(y, n_dims=None):
    """ Take integer y (tensor or variable) with n dims and convert it to 1-hot representation with n+1 dims. """
    y_tensor  = y.data
    y_tensor  = y_tensor.type(torch.LongTensor).view(-1, 1)
    n_dims    = n_dims if n_dims is not None else int(torch.max(y_tensor)) + 1
    y_one_hot = torch.zeros(y_tensor.size()[0], n_dims).scatter_(1, y_tensor, 1)
    y_one_hot = y_one_hot.view(*y.shape, -1)
    y_one_hot = y_one_hot.transpose(-1, 1).transpose(-1, 2)
    return y_one_hot

def dice_loss(input,target,nclasses,use_gpu=True):
    """
    input is a torch variable of size BatchxnclassesxHxW representing log probabilities for each class
    target is of the groundtruth, shoud have same size as the input
    """
    if use_gpu:
        target = target.cuda()
    probs = F.sigmoid(input)

    num   = (probs*target).sum() + 1e-3
    den   = probs.sum() + target.sum() + 1e-3
    dice  = 2.*(num/den)
    return 1. - dice


class Struct:
    def __init__(self, **entries):
        self.__dict__.update(entries)