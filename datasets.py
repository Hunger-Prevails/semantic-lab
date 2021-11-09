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
        self.data_name = args.data_name
        self.n_classes = args.n_classes

        with open('metadata.json') as file:
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

        return samples


    def parse_sample(self, sample):
        image = np.flip(cv2.imread(sample.image_path), axis = -1).copy()
        label = np.load(sample.label_path)

        image = self.transforms(augmentation.random_colour(image) if self.enc_colour else image)

        return image, label


    def __getitem__(self, index):
        return self.parse_sample(self.samples[index])


    def __len__(self):
        return len(self.samples)
