import os
import sys
import cv2
import json
import pickle
import random
import numpy as np
import torch
import torch.utils.data as data

import augmentation


def get_loader(args, phase):
    dataset = Dataset(args, phase)

    shuffle = args.shuffle if phase == 'train' else False

    batch_size = args.batch_size if phase == 'train' else args.batch_size + args.batch_size

    return data.DataLoader(dataset, batch_size, shuffle, num_workers = args.n_workers, pin_memory = True)


def to_tensor(array):
    tensor = torch.from_numpy(array)
    tensor = tensor.float().div(255.0)

    return tensor.permute(2, 0, 1)


class Dataset(data.Dataset):

    def __init__(self, args, phase):
        self.data_name = args.data_name
        self.n_classes = args.n_classes
        self.attention = args.attention and phase == 'train'

        with open('/home/yinglun.liu/Datasets/metadata.json') as file:
            metadata = json.load(file)

        self.samples = []
        for data_name in args.data_name:
            n_samples = getattr(self, 'get_' + data_name + '_samples')(metadata['root'][data_name], phase)

            print('=> => collects [ {:d} ] {:s} samples from {:s}'.format(n_samples, phase, data_name))

        transforms = []

        if args.augment_rand_crop and phase == 'train':
            transforms += [augmentation.RandCrop(args.n_steps_rand_crop, [320, 480])]

        if args.augment_tint_sway and phase == 'train':
            transforms += [augmentation.TintSway(0.05)]

        transforms += [augmentation.Normalize(metadata['mean'], metadata['stddvn'])]

        self.transforms = augmentation.Compose(transforms)
        assert self.__len__() != 0


    def get_smile_view_samples(self, root, phase):
        with open(os.path.join(root, phase + '.pkl'), 'rb') as file:
            samples = pickle.load(file)

        if phase != 'train':
            samples = [sample for sample in samples if not sample.to_flip]

        self.samples += samples
        return len(samples)


    def get_smile_architect_samples(self, root, phase):
        with open(os.path.join(root, phase + '.pkl'), 'rb') as file:
            samples = pickle.load(file)

        if phase != 'train':
            samples = [sample for sample in samples if not sample.to_flip]

        self.samples += samples
        return len(samples)


    def to_atten(self, label):
        atten = np.ones(label.shape).astype(np.float32)

        nontrivia = np.logical_and(0 < label, label < self.n_classes)
        cnt_atten = nontrivia.sum()

        atten[nontrivia] = (label.size - cnt_atten) / cnt_atten
        atten = np.multiply(atten, label.size / atten.sum())

        return cv2.filter2D(atten, -1, np.ones((20, 20)) / 400)


    def parse_sample(self, sample):
        image = sample.read()
        label = np.load(sample.label_path)

        ret = dict(image = to_tensor(image), label = torch.from_numpy(label))
        if self.attention:
            ret['atten'] = torch.from_numpy(self.to_atten(label))

        self.transforms(ret)
        return ret


    def __getitem__(self, index):
        return self.parse_sample(self.samples[index])


    def __len__(self):
        return len(self.samples)
