import torch
import numpy as np
import random

from torchvision.transforms import functional as F
from torchvision.transforms.functional import InterpolationMode as Mode


class BatchCrop(object):
    """
    Function:
        Applies torchvision.transforms.RandomCrop transform to a batch of images and associated labels.
        Makes sure images and labels undergo the same transformation.
    Args:
        dest_shape (int, int): desired output shape of the crop
    """
    def __init__(self, n_steps):
        self.crop_rates = np.linspace(1.0 - np.multiply(n_steps - 1, 0.05), 1.0, num = n_steps)

    def __call__(self, batch):
        """
        Args:
            tensor (Tensor): Tensor of size (N, C, H, W) to be cropped
            labels (Tensor): Labels of size (N, H, W) to be cropped
        Returns:
            Tensor: Randomly cropped tensor
            Tensor: Randomly cropped tensor
        """
        crop_rate = random.choice(self.crop_rates)

        h_srce, w_srce = batch['image'].shape[2:]
        h_dest, w_dest = int(torch.mul(h_srce, crop_rate)), int(torch.mul(w_srce, crop_rate))

        indices = torch.arange(batch['image'].shape[0])[:, None, None]

        i = torch.randint(0, h_srce - h_dest + 1, batch['image'].shape[:1])
        j = torch.randint(0, w_srce - w_dest + 1, batch['image'].shape[:1])

        rows = torch.arange(h_dest, dtype = torch.long) + i[:, None]
        cols = torch.arange(w_dest, dtype = torch.long) + j[:, None]

        tensor = batch['image'].permute(1, 0, 2, 3)
        tensor = tensor[:, indices, rows[:, torch.arange(h_dest)[:, None]], cols[:, None]]

        batch['image'] = tensor.permute(1, 0, 2, 3)
        batch['label'] = batch['label'][indices, rows[:, torch.arange(h_dest)[:, None]], cols[:, None]]

        if 'atten' in batch:
            batch['atten'] = batch['atten'][indices, rows[:, torch.arange(h_dest)[:, None]], cols[:, None]]


class RandCrop(object):
    """
    Function:
        Applies torchvision.transforms.RandomCrop transform to a single image and its associated label.
        Makes sure images and labels undergo the same transformation.
    Args:
        dest_shape (int, int): desired output shape of the crop
    """
    def __init__(self, n_steps, dimensions):
        self.dimensions = dimensions
        self.crop_rates = np.linspace(1.0 - np.multiply(n_steps - 1, 0.05), 1.0, num = n_steps)


    def __call__(self, ret):
        crop_rate = random.choice(self.crop_rates)

        h_srce, w_srce = ret['image'].shape[1:]
        h_dest, w_dest = int(torch.mul(h_srce, crop_rate)), int(torch.mul(w_srce, crop_rate))

        t_dest = np.random.randint(h_srce - h_dest + 1)
        l_dest = np.random.randint(w_srce - w_dest + 1)

        ret['image'] = F.crop(ret['image'], t_dest, l_dest, h_dest, w_dest)
        ret['image'] = F.resize(ret['image'], self.dimensions)

        ret['label'] = F.crop(ret['label'], t_dest, l_dest, h_dest, w_dest)
        ret['label'] = F.resize(ret['label'][None], self.dimensions, Mode('nearest'))
        ret['label'] = ret['label'].squeeze()

        if 'atten' in ret:
            ret['atten'] = F.crop(ret['atten'], t_dest, l_dest, h_dest, w_dest)
            ret['atten'] = F.resize(ret['atten'][None], self.dimensions)
            ret['atten'] = ret['atten'].squeeze()
