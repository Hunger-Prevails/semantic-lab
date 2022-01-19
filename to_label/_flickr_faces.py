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


def to_sample(root, phase, image_name, annotation, mirror):
    print('handles', image_name)

    image_path = os.path.join(root, phase, image_name)
    label_path = os.path.join(root, phase, image_name.replace('.png', '.marking.png'))
    final_path = os.path.join(root, phase + '_crop', image_name)
    
    sample_a = _utils.Sample(final_path, final_path.replace('.png', '.marking.0.npy'))
    sample_b = _utils.Sample(final_path, final_path.replace('.png', '.marking.1.npy'), to_flip = True)

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

        cv2.imwrite(final_path, np.flip(dest_image, axis = -1))

        np.save(sample_a.label_path, dest_label)

    if not os.path.exists(sample_b.label_path):
        dest_label = np.load(sample_a.label_path)

        np.save(sample_b.label_path, _utils.flip(dest_label, mirror))

    return [sample_a, sample_b]


def gather(phase, root, annotation, mirror):
    print('gathers', phase, 'samples')

    if not os.path.exists(os.path.join(root, phase + '_crop')):
        os.mkdir(os.path.join(root, phase + '_crop'))

    image_files = os.listdir(os.path.join(root, phase))
    image_files.sort()
    image_files = image_files[1::2]

    processes = list()
    pool = multiprocessing.Pool(6)

    for image_file in image_files:
        processes += [pool.apply_async(func = to_sample, args = (root, phase, image_file, annotation, mirror))]

    pool.close()
    pool.join()

    samples = sum([process.get() for process in processes], list())

    print('collects', len(samples), phase, 'samples')

    with open(os.path.join(root, phase + '.pkl'), 'wb') as file:
        file.write(pickle.dumps(samples))


def main():
    with open('/home/yinglun.liu/Datasets/metadata.json') as file:
        metadata = json.load(file)

    root = metadata['root']['flickr_faces']

    annotation = metadata['annotation']['flickr_faces']
    annotation = np.array(annotation)

    mirror = np.array(metadata['mirror']['flickr_faces'])
    mirror = dict(np.vstack([mirror, np.flip(mirror, axis = -1)]).tolist())

    gather('test', root, annotation, mirror)


if __name__ == '__main__':
    main()
