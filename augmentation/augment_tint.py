import cv2
import numpy as np
import random

from functools import partial

from torchvision.transforms import functional as F


def augment_brightness(image, space):
    if space != 'rgb':
        image = cv2.cvtColor(image, cv2.COLOR_HSV2RGB)

    image += np.random.uniform(- 0.2, 0.2)

    return np.clip(image, 0, 1), 'rgb'


def augment_contrast(image, space):
    if space != 'rgb':
        image = cv2.cvtColor(image, cv2.COLOR_HSV2RGB)

    image -= 0.5
    image /= np.random.uniform(4 / 5, 5 / 4)
    image += 0.5
    
    return np.clip(image, 0, 1), 'rgb'


def augment_hue(image, space):
    if space != 'hsv':
        image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    image[:, :, 0] += np.random.uniform(- 18, 18)

    image[:, :, 0][image[:, :, 0] < 0] += 360
    image[:, :, 0][360 <= image[:, :, 0]] -= 360
    
    return image, 'hsv'


def augment_saturation(image, space):
    if space != 'hsv':
        image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    image[:, :, 1] /= np.random.uniform(4 / 5, 5 / 4)
    image[:, :, 1] = np.clip(image[:, :, 1], 0, 1)

    return image, 'hsv'


def augment_tint(image):
    '''
    performs random colour augmentation for the given image

    Args:
        image: 3-channel rgb image in range [0, 256)
    '''
    dest = (image / 255.0).astype(np.float32)

    augment_funcs = [augment_brightness, augment_contrast, augment_hue, augment_saturation]
    random.shuffle(augment_funcs)

    colorspace = 'rgb'

    for func in augment_funcs:
        dest, colorspace = func(dest, colorspace)

    if colorspace != 'rgb':
        dest = cv2.cvtColor(dest, cv2.COLOR_HSV2RGB)

    return np.multiply(dest, 255).astype(np.uint8)


class TintSway(object):
    def __init__(self, max_hue):
        self.functor = dict()
        self.functor['hue'] = F.adjust_hue
        self.functor['contrast'] = F.adjust_contrast
        self.functor['saturation'] = F.adjust_saturation
        self.functor['brightness'] = lambda image, shift: image + shift

        self.params = dict()
        self.params['hue'] = [- max_hue, max_hue]
        self.params['contrast'] = [4 / 5, 5 / 4]
        self.params['saturation'] = [4 / 5, 5 / 4]
        self.params['brightness'] = [- 1 / 5, 1 / 5]

        self.names = ['hue', 'contrast', 'saturation', 'brightness']


    def apply(self, name, ret):
        param = np.random.uniform(self.params[name][0], self.params[name][1])
        ret['image'] = self.functor[name](ret['image'], param)


    def __call__(self, ret):
        random.shuffle(self.names)
        map(partial(self.apply, ret = ret), self.names)
