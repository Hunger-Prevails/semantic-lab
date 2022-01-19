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

    root = metadata['root']['smile_view']

    annotation = metadata['annotation']['smile_view']
    annotation = np.array(annotation)

    mirror = np.array(metadata['mirror']['smile_view'])
    mirror = dict(np.vstack([mirror, np.flip(mirror, axis = -1)]).tolist())

    gather(annotation, mirror, root, 'test')
    gather(annotation, mirror, root, 'train')
    gather(annotation, mirror, root, 'validation')


def gather(annotation, mirror, root, phase):
    print('gathers', phase, 'samples')

    files = glob.glob(os.path.join(root, phase, '*.png'))
    files.sort()

    label_files = files[0::2]
    image_files = files[1::2]

    samples = list()

    for image_file, label_file in zip(image_files, label_files):
        sample_a = _utils.Sample(image_file, label_file.replace('.0.png', '.0.npy'))
        sample_b = _utils.Sample(image_file, label_file.replace('.0.png', '.1.npy'), to_flip = True)

        samples += [sample_a, sample_b]

        if not os.path.exists(sample_a.label_path):
            np.save(sample_a.label_path, _utils.parse_image(label_file, annotation))

        if not os.path.exists(sample_b.label_path):
            np.save(sample_b.label_path, _utils.flip(np.load(sample_a.label_path), mirror))

    with open(os.path.join(root, phase + '.pkl'), 'wb') as file:
        file.write(pickle.dumps(samples))


if __name__ == '__main__':
    main()