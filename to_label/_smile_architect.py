import os
import sys
import cv2
import glob
import json
import numpy as np
import pickle
import random
import multiprocessing

from . import _utils


def to_sample(phase, image_path, label_path, annotation, mirror):
    print('=> handles', os.path.basename(image_path), 'and', os.path.basename(label_path))

    dest_image_path = image_path.replace(phase, phase + '_crop')
    dest_label_path = label_path.replace(phase, phase + '_crop')

    sample_a = _utils.Sample(dest_image_path, dest_label_path.replace('.png', '.npy'))
    sample_b = _utils.Sample(dest_image_path, dest_label_path.replace('.0.png', '.1.npy'), to_flip = True)

    if not os.path.exists(sample_a.label_path):
        image = cv2.imread(image_path)
        label = cv2.imread(label_path)

        crop_box = _utils.crop_box(label, annotation[0])

        crop_image = image[crop_box[1]:crop_box[1] + crop_box[3], crop_box[0]:crop_box[0] + crop_box[2]]
        crop_label = label[crop_box[1]:crop_box[1] + crop_box[3], crop_box[0]:crop_box[0] + crop_box[2]]

        crop_image = np.flip(crop_image, axis = -1).copy()
        crop_label = np.flip(crop_label, axis = -1).copy()

        crop_label = _utils.to_label(crop_label, annotation)

        dest_image = cv2.resize(crop_image, (480, 320))
        dest_label = cv2.resize(crop_label, (480, 320), interpolation = cv2.INTER_NEAREST)

        cv2.imwrite(dest_image_path, np.flip(dest_image, axis = -1))

        np.save(sample_a.label_path, dest_label)

    if not os.path.exists(sample_b.label_path):
        dest_label = np.load(sample_a.label_path)

        np.save(sample_b.label_path, _utils.flip(dest_label, mirror))

    return [sample_a, sample_b]


def gather(phase, root, annotation, mirror):
    print('gathers', phase, 'samples')

    if not os.path.exists(os.path.join(root, phase + '_crop')):
        os.mkdir(os.path.join(root, phase + '_crop'))

    files = glob.glob(os.path.join(root, phase, '*.png'))
    files.sort()

    label_files = files[0::2]
    image_files = files[1::2]

    processes = list()
    pool = multiprocessing.Pool(6)

    for image_file, label_file in zip(image_files, label_files):
        processes += [pool.apply_async(func = to_sample, args = (phase, image_file, label_file, annotation, mirror))]

    pool.close()
    pool.join()

    samples = sum([process.get() for process in processes], list())

    print('collects', len(samples), phase, 'samples')

    with open(os.path.join(root, phase + '.pkl'), 'wb') as file:
        file.write(pickle.dumps(samples))


def main():
    with open('/home/yinglun.liu/Datasets/metadata.json') as file:
        metadata = json.load(file)

    root = metadata['root']['smile_architect']

    annotation = metadata['annotation']['smile_architect']
    annotation = np.stack(annotation)

    mirror = np.array(metadata['mirror']['smile_architect'])
    mirror = dict(np.vstack([mirror, np.flip(mirror, axis = -1)]).tolist())

    gather('test', root, annotation, mirror)
    gather('train', root, annotation, mirror)
    gather('validation', root, annotation, mirror)


if __name__ == '__main__':
    main()
