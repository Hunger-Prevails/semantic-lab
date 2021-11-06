import torch
import numpy as np
import random


class RandomCrop:
    """
    Function:
        Applies torchvision.transforms.RandomCrop transform to a batch of images and associated labels.
        Makes sure images and labels undergo the same transformation.
    Args:
        dest_shape (int, int): desired output shape of the crop
    """
    def __init__(self, crop_rate):
        self.crop_rates = np.linspace(crop_rate, 1.0, num = 5)


    def __call__(self, tensor, labels):
        """
        Args:
            tensor (Tensor): Tensor of size (N, C, H, W) to be cropped
            labels (Tensor): Labels of size (N, H, W) to be cropped
        Returns:
            Tensor: Randomly cropped tensor
            Tensor: Randomly cropped tensor
        """
        crop_rate = random.choice(self.crop_rates)

        h_size, w_size = tensor.shape[2:]
        h_dest, w_dest = int(h_size * crop_rate), int(w_size * crop_rate)

        i = torch.randint(0, h_size - h_dest + 1, (tensor.size(0),))
        j = torch.randint(0, w_size - w_dest + 1, (tensor.size(0),))

        rows = torch.arange(h_dest, dtype = torch.long) + i[:, None]
        columns = torch.arange(w_dest, dtype = torch.long) + j[:, None]

        new_tensor = tensor.permute(1, 0, 2, 3)
        new_tensor = new_tensor[:, torch.arange(tensor.size(0))[:, None, None], rows[:, torch.arange(h_dest)[:, None]], columns[:, None]]
        new_labels = labels[torch.arange(tensor.size(0))[:, None, None], rows[:, torch.arange(h_dest)[:, None]], columns[:, None]]
        return new_tensor.permute(1, 0, 2, 3), new_labels
