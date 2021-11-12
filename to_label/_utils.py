import os
import sys
import cv2
import numpy as np

class Sample:
    def __init__(self, image_path, label_path, to_flip = False):
        try:
            assert os.path.basename(image_path).split('.')[1] == 'orig'
            assert os.path.basename(label_path).split('.')[1] == 'marking'

            i_image = int(os.path.basename(image_path).split('.')[0])
            i_label = int(os.path.basename(label_path).split('.')[0])
            assert i_image == i_label
        except:
            print('file mismatch:', image_path, label_path)
            sys.exit(0)

        self.to_flip = to_flip
        self.image_path = image_path
        self.label_path = label_path

    def read(self):
        image = np.flip(cv2.imread(self.image_path), axis = -1)

        return np.flip(image, axis = 1).copy() if self.to_flip else image.copy()
