import os
import cv2
import json
import numpy as np
import random
import torch
import glob
import torch.utils.data as data

from torchvision import transforms


def get_loader(args, phase):
    dataset = Dataset(args, phase)

    shuffle = args.shuffle if phase == 'train' else False

    return data.DataLoader(dataset, args.batch_size, shuffle, num_workers = args.n_workers, pin_memory = True)


class Sample:
    def __init__(self, image_path, label_path):
        self.image_path = image_path
        self.label_path = label_path


class Dataset(data.Dataset):

    def __init__(self, args, phase):
        self.phase = phase
        self.data_name = args.data_name
        self.n_classes = args.n_classes

        with open('metadata.json') as file:
            metadata = json.load(file)

        self.root = metadata['root'][args.data_name]

        annotation = metadata['annotation'][args.data_name]

        self.label_map = dict([(tuple(anno), i) for i, anno in enumerate(annotation)])

        self.samples = getattr(self, 'get_' + args.data_name + '_samples')(phase)

        self.transforms = [
            transforms.ToTensor(),
            transforms.Normalize(mean = metadata['mean'], std = metadata['stddvn'])
        ]
        self.transforms = transforms.Compose(self.transforms)

        print('\tcollects [', self.__len__(), ']', phase, 'samples')


    def get_smile_view_samples(self, phase):
        files = glob.glob(os.path.join(self.root, self.phase, '*.png'))
        files.sort()

        image_files = files[0::2]
        label_files = files[1::2]

        return [Sample(image_file, label_file) for image_file, label_file in zip(image_files, label_files)]


    def parse_sample(self, sample):
        image = cv2.imread(sample.image_path)
        label_image = cv2.imread(sample.label_path)

        image = self.transforms(random_color(image) if self.colour else image)
        label = self.to_label(label_image)

        return image, label


    def to_label(self, label_image):
        shape = label_image.shape[:2]

        label_image = label_image.reshape(-1, 3)

        label = np.array([self.label_map[tuple(anno)] for anno in label_image])

        label = np.expand_dims(label.reshape(shape), -1)

        return np.put_along_axis(np.zeros(shape + [self.n_classes]), label, 1, -1)


    def __getitem__(self, index):
        return self.parse_sample(self.samples[index])


    def __len__(self):
        return len(self.samples)
