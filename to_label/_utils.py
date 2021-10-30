import os
import sys

class Sample:
    def __init__(self, image_path, label_path):
        try:
            assert os.path.basename(image_path).split('.')[1] == 'orig'
            assert os.path.basename(label_path).split('.')[1] == 'marking'

            i_image = int(os.path.basename(image_path).split('.')[0])
            i_label = int(os.path.basename(label_path).split('.')[0])
            assert i_image == i_label
        except:
            print('file mismatch:', image_path, label_path)
            sys.exit(0)

        self.image_path = image_path
        self.label_path = label_path
