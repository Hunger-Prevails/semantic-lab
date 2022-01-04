import os
import sys
import cv2
import glob
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from functools import partial

import utils
import flood
import models

from models import FakeArgs
from exceptions import BoxException

np.seterr(all = 'raise')

class Anonymizer(object):

	def __init__(self, model, metadata):
		self.model = model
		self.metadata = metadata


	def save_fig(self, save_path, srce_image, face_bbox, jaws_bbox, jaws_image, jaws_label, y_coord = None):
		plt.figure(figsize = (16, 8))

		Rectangle = partial(patches.Rectangle, linewidth = 2, facecolor = 'none')
		annotation = self.metadata['annotation']['smile_view']

		ax = plt.subplot(1, 4, 1)
		ax.imshow(srce_image)
		rect = Rectangle((face_bbox[0], face_bbox[1]), face_bbox[2], face_bbox[3], edgecolor = 'r')
		ax.add_patch(rect)
		rect = Rectangle((jaws_bbox[0], jaws_bbox[1]), jaws_bbox[2], jaws_bbox[3], edgecolor = 'b')
		ax.add_patch(rect)

		ax = plt.subplot(1, 4, 2)
		ax.imshow(jaws_image)

		ax = plt.subplot(1, 4, 3)
		ax.imshow(utils.to_label_image(jaws_label, annotation))

		if y_coord is not None:
			ax = plt.subplot(1, 4, 4)
			ax.imshow(srce_image)
			ax.axhline(y = y_coord, color = 'g')

		plt.savefig(save_path)
		plt.close()


	def anonymize(self, image_path):
		image_name = os.path.basename(image_path)
		figur_name = image_name.replace('.png', '.jpg')
		label_name = image_name.replace('.png', '.npy')

		if os.path.exists(os.path.join(self.dest_path, figur_name)):
			return
		print('=> => anonymizes image:', image_name)

		srce_image = np.flip(cv2.imread(image_path), axis = -1).copy()
		try:
			face_bbox, jaws_bbox, face_marks = utils.detect_jaws(srce_image)
		except BoxException as exception:
			print('=> => => {:s} on {:s}'.format(exception.message, image_path))
			print('=> => => copies image to', self.fail_path['detect'])

			if not os.path.exists(self.fail_path['detect']):
				os.mkdir(self.fail_path['detect'])
			cv2.imwrite(os.path.join(self.fail_path['detect'], image_name), np.flip(srce_image, axis = -1))
			return

		jaws_image = utils.crop_jaws(srce_image, jaws_bbox)
		jaws_label = utils.detect_teeth(jaws_image, self.model, self.metadata)
		annotation = self.metadata['annotation']['smile_view']
		try:
			y_coord = utils.get_y_coord(jaws_bbox, flood.ransac(jaws_label), len(annotation))
		except:
			print('=> => => smodel fails on', image_path)
			print('=> => => copies image to', self.fail_path['smodel'])

			if not os.path.exists(self.fail_path['smodel']):
				os.mkdir(self.fail_path['smodel'])
			cv2.imwrite(os.path.join(self.fail_path['smodel'], image_name), np.flip(srce_image, axis = -1))
			np.save(os.path.join(self.fail_path['smodel'], label_name), jaws_label)
			save_path = os.path.join(self.fail_path['smodel'], figur_name)
			self.save_fig(save_path, srce_image, face_bbox, jaws_bbox, jaws_image, jaws_label)
			return

		anonym_crop = np.flip(srce_image[int(y_coord):], axis = -1)
		cv2.imwrite(os.path.join(self.dest_path, image_name), anonym_crop)
		landmarks = os.path.join(self.dest_path, image_name.replace('.png', '.landmarks.json'))

		with open(landmarks, 'w') as file:
			file.write(json.dumps(face_marks))


	def run(self, srce_path, dest_path):
		self.srce_path = srce_path
		self.dest_path = dest_path
		self.fail_path = dict(
			detect = os.path.join(dest_path, 'detect_failure'),
			smodel = os.path.join(dest_path, 'smodel_failure')
		)
		print('=> anonimization starts')

		image_paths = glob.glob(os.path.join(srce_path, '*.png'))
		image_paths.sort()
		for image_path in image_paths:
			self.anonymize(image_path)

		print('<= anonimization finishes')


def main(model_path, srce_path, dest_path):
	model = models.create_model(FakeArgs(), model_path)

	with open('res/metadata.json') as file:
		metadata = json.load(file)

	anonymizer = Anonymizer(model, metadata)
	anonymizer.run(srce_path, dest_path)


if __name__ == '__main__':
	main(sys.argv[1], sys.argv[2], sys.argv[3])
