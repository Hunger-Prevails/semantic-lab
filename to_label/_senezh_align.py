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


def to_sample(phase, index, image_path, annotation, mirror):
    print('handles', os.path.basename(image_path))

    label_path = image_path.replace('.png', '_seg.png')

    dest_path = os.path.dirname(image_path).replace(phase, phase + '_crop')

    dest_image_name = str(index).zfill(4) + '.orig.png'
    dest_label_name = str(index).zfill(4) + '.marking.png'

    dest_image_path = os.path.join(dest_path, dest_image_name)
    dest_label_path = os.path.join(dest_path, dest_label_name)

    sample_a = _utils.Sample(dest_image_path, dest_label_path.replace('.png', '.0.npy'))
    sample_b = _utils.Sample(dest_image_path, dest_label_path.replace('.png', '.1.npy'), to_flip = True)

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

    image_files = files[0::2]

    processes = list()
    pool = multiprocessing.Pool(6)

    for index, image_file in enumerate(image_files):
        processes += [pool.apply_async(func = to_sample, args = (phase, index, image_file, annotation, mirror))]

    pool.close()
    pool.join()

    samples = sum([process.get() for process in processes], list())

    print('collects', len(samples), phase, 'samples')

    with open(os.path.join(root, phase + '.pkl'), 'wb') as file:
        file.write(pickle.dumps(samples))


def main():
    with open('/home/yinglun.liu/Datasets/metadata.json') as file:
        metadata = json.load(file)

    root = metadata['root']['senezh_align']

    annotation = metadata['annotation']['senezh_align']
    annotation = np.array(annotation)

    mirror = np.array(metadata['mirror']['senezh_align'])
    mirror = dict(np.vstack([mirror, np.flip(mirror, axis = -1)]).tolist())

    gather('test', root, annotation, mirror)


if __name__ == '__main__':
    main()
