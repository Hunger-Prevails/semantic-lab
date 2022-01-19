import os
import cv2
import glob
import json
import pickle
import numpy as np

from . import _utils


def main():
    with open('/home/yinglun.liu/Datasets/metadata.json') as file:
        metadata = json.load(file)

    root = metadata['root']['senezh_exocad']

    annotation = metadata['annotation']['senezh_exocad']
    annotation = np.array(annotation)

    mirror = np.array(metadata['mirror']['senezh_exocad'])
    mirror = dict(np.vstack([mirror, np.flip(mirror, axis = -1)]).tolist())

    gather(annotation, mirror, root, 'test')


def gather(annotation, mirror, root, phase):
    print('gathers', phase, 'samples')

    files = glob.glob(os.path.join(root, phase, '*.png'))
    files.sort()

    image_files = files[0::2]
    label_files = files[1::2]

    samples = list()

    for image_file, label_file in zip(image_files, label_files):
        sample_a = _utils.Sample(image_file, label_file.replace('.png', '.marking.0.npy'))
        sample_b = _utils.Sample(image_file, label_file.replace('.png', '.marking.1.npy'), to_flip = True)

        samples += [sample_a, sample_b]

        if not os.path.exists(sample_a.label_path):
            np.save(sample_a.label_path, _utils.parse_image(label_file, annotation))

        if not os.path.exists(sample_b.label_path):
            np.save(sample_b.label_path, _utils.flip(np.load(sample_a.label_path), mirror))

    with open(os.path.join(root, phase + '.pkl'), 'wb') as file:
        file.write(pickle.dumps(samples))


if __name__ == '__main__':
    main()