import os
import sys
import cv2
import json
import pickle
import random
import numpy as np
import torch
import torch.utils.data as data

from torchvision import transforms

import augmentation


def get_loader(args, phase):
    dataset = Dataset(args, phase)

    shuffle = args.shuffle if phase == 'train' else False

    batch_size = args.batch_size if phase == 'train' else args.batch_size + args.batch_size

    return data.DataLoader(dataset, batch_size, shuffle, num_workers = args.n_workers, pin_memory = True)


class Dataset(data.Dataset):

    def __init__(self, args, phase):
        self.phase = phase
        self.data_name = args.data_name
        self.attention = args.attention
        self.n_classes = args.n_classes

        with open('/home/yinglun.liu/Datasets/metadata.json') as file:
            metadata = json.load(file)

        self.root = metadata['root'][args.data_name]

        self.samples = getattr(self, 'get_' + args.data_name + '_samples')(phase)

        self.transforms = [
            transforms.ToTensor(),
            transforms.Normalize(mean = metadata['mean'], std = metadata['stddvn'])
        ]
        self.transforms = transforms.Compose(self.transforms)

        self.enc_colour = args.enc_colour and phase == 'train'

        print('=> => collects [', self.__len__(), ']', phase, 'samples')


    def get_smile_view_samples(self, phase):
        with open(os.path.join(self.root, phase + '.pkl'), 'rb') as file:
            samples = pickle.load(file)

        if phase != 'train':
            samples = [sample for sample in samples if not sample.to_flip]

        return samples


    def to_atten(self, label):
        atten = np.ones(label.shape).astype(np.float32)

        cnt_atten = (0 < label).sum()

        atten[0 < label] = (label.size - cnt_atten) / cnt_atten
        atten = np.multiply(atten, label.size / atten.sum())

        return cv2.filter2D(atten, -1, np.ones((20, 20)) / 400)


    def parse_sample(self, sample):
        image = sample.read()
        label = np.load(sample.label_path)

        image = self.transforms(augmentation.random_colour(image) if self.enc_colour else image)

        ret = dict(image = image, label = label)

        if self.attention and self.phase == 'train':
            ret['atten'] = self.to_atten(label)

        return ret


    def __getitem__(self, index):
        return self.parse_sample(self.samples[index])


    def __len__(self):
        return len(self.samples)
