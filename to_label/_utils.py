import os
import sys
import cv2
import numpy as np

class Sample:
    def __init__(self, image_path, label_path, to_flip = False):
        try:
            image_name = os.path.basename(image_path)
            label_name = os.path.basename(label_path)

            assert image_name.split('.')[1] == 'orig'
            assert label_name.split('.')[1] == 'marking'
            assert image_name.split('.')[0] == label_name.split('.')[0]
        except:
            print('file mismatch:', image_path, label_path)
            sys.exit(0)

        self.to_flip = to_flip
        self.image_path = image_path
        self.label_path = label_path

    def read(self):
        image = np.flip(cv2.imread(self.image_path), axis = -1)

        return np.flip(image, axis = 1).copy() if self.to_flip else image.copy()


def to_label(label_image, label_map):
    shape = label_image.shape[:2]

    label = np.array([label_map.get(tuple(pixel), len(label_map)) for pixel in label_image.reshape(-1, 3)])

    return label.reshape(shape).astype(np.uint8)


def to_label_image(label, annotation):
    dest_shape = list(label.shape) + [3]

    label_image = np.array([annotation[l] for l in label.flatten()])

    return label_image.reshape(dest_shape).astype(np.uint8)


def flip(label, mirror):
    shape = label.shape[:2]

    label = np.flip(label, axis = -1).copy().flatten()

    m_label = np.array([mirror.get(l, l) for l in label])

    return m_label.reshape(shape)


def crop_box(label, n_classes):
    h_coords, w_coords = np.where(np.logical_and(0 < label, label < n_classes))

    t_bound, b_bound = np.amin(h_coords), np.amax(h_coords)
    l_bound, r_bound = np.amin(w_coords), np.amax(w_coords)

    box_h = np.multiply((b_bound - t_bound), 10) // 3
    box_w = np.multiply((r_bound - l_bound), 3) // 2

    box_h = max(box_h, np.multiply(box_w, 2) // 3)
    box_w = max(box_w, np.multiply(box_h, 3) // 2)

    box_t = (t_bound + b_bound) // 2 - box_h // 2
    box_l = (l_bound + r_bound) // 2 - box_w // 2

    return np.stack([box_l, box_t, box_w, box_h])
