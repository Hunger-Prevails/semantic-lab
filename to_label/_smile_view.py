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

    mirror = np.array(metadata['mirror']['smile_view'])
    mirror = dict(np.vstack([mirror, np.flip(mirror, axis = -1)]).tolist())

    label_map = dict([(tuple(anno), i) for i, anno in enumerate(annotation)])

    gather(label_map, mirror, root, 'test')
    gather(label_map, mirror, root, 'train')
    gather(label_map, mirror, root, 'validation')


def gather(label_map, mirror, root, phase):
    print('gathers', phase, 'samples')

    files = glob.glob(os.path.join(root, phase, '*.png'))
    files.sort()

    label_files = files[0::2]
    image_files = files[1::2]

    samples = list()

    for image_file, label_file in zip(image_files, label_files):
        sample = _utils.Sample(image_file, label_file.replace('.0.png', '.0.npy'))
        
        if not os.path.exists(sample.label_path):
            label = _utils.parse_image(label_file, label_map)
            np.save(sample.label_path, label)

        samples.append(sample)
        m_sample = _utils.Sample(image_file, label_file.replace('.0.png', '.1.npy'), to_flip = True)

        if not os.path.exists(m_sample.label_path):
            label = np.load(sample.label_path)
            m_label = _utils.flip(label, mirror)
            np.save(m_sample.label_path, m_label)

        samples.append(m_sample)

    with open(os.path.join(root, phase + '.pkl'), 'wb') as file:
        file.write(pickle.dumps(samples))


if __name__ == '__main__':
    main()