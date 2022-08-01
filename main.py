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
import time

from models import FakeArgs
from exceptions import BoxException

np.seterr(all = 'raise')

class Anonymizer(object):

	def __init__(self, model, metadata):
		self.model = model
		self.metadata = metadata
		self.total_elapse = 0
		self.total_number = 0


	def as_flow(self, save_path, srce_image, face_bbox, jaws_bbox, jaws_image, jaws_label):
		plt.figure(figsize = (16, 8))

		Rectangle = partial(patches.Rectangle, linewidth = 2, facecolor = 'none')
		annotation = self.metadata['annotation']['flickr_faces']

		ax = plt.subplot(1, 3, 1)
		ax.axis('off')
		ax.imshow(srce_image)
		rect = Rectangle((face_bbox[0], face_bbox[1]), face_bbox[2], face_bbox[3], edgecolor = 'r')
		ax.add_patch(rect)
		rect = Rectangle((jaws_bbox[0], jaws_bbox[1]), jaws_bbox[2], jaws_bbox[3], edgecolor = 'b')
		ax.add_patch(rect)

		ax = plt.subplot(1, 3, 2)
		ax.axis('off')
		ax.imshow(jaws_image)

		ax = plt.subplot(1, 3, 3)
		ax.axis('off')
		ax.imshow(utils.to_label_image(jaws_label, annotation))

		plt.savefig(save_path)
		plt.close()


	def as_dest(self, srce_image, y_coord, face_marks):
		dest_image = srce_image.copy()
		dest_image[:y_coord] = 0

		with open('res/colormap.json') as file:
			colormap = json.load(file)

		for key in face_marks:
			landmarks = np.stack(face_marks[key])
			landmarks = landmarks[landmarks[:, 1] < y_coord]

			for landmark in landmarks:
				cv2.circle(dest_image, tuple(landmark), 4, colormap[key], cv2.FILLED)

		return dest_image


	def anonymize(self, image_path):
		image_name = os.path.basename(image_path)
		figur_name = image_name.replace('.png', '.jpg')
		label_name = image_name.replace('.png', '.npy')
		marks_name = image_name.replace('.png', '.landmarks.json')

		if os.path.exists(os.path.join(self.dest_path, image_name)):
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

		time_b = time.time()

		jaws_label = utils.detect_teeth(jaws_image, self.model, self.metadata)

		time_e = time.time()

		self.total_elapse += time_e - time_b
		self.total_number += 1

		try:
			y_coord = utils.get_y_coord(jaws_bbox, flood.ransac(jaws_label))
		except:
			print('=> => => smodel fails on', image_path)
			print('=> => => copies image to', self.fail_path['smodel'])

			if not os.path.exists(self.fail_path['smodel']):
				os.mkdir(self.fail_path['smodel'])
			cv2.imwrite(os.path.join(self.fail_path['smodel'], image_name), np.flip(srce_image, axis = -1))
			np.save(os.path.join(self.fail_path['smodel'], label_name), jaws_label)
			save_path = os.path.join(self.fail_path['smodel'], figur_name)
			self.as_flow(save_path, srce_image, face_bbox, jaws_bbox, jaws_image, jaws_label)
			return

		with open(os.path.join(self.dest_path, marks_name), 'w') as file:
			file.write(json.dumps(face_marks))

		save_path = os.path.join(self.dest_path, figur_name)
		self.as_flow(save_path, srce_image, face_bbox, jaws_bbox, jaws_image, jaws_label)
		dest_image = self.as_dest(srce_image, int(y_coord), face_marks)
		cv2.imwrite(os.path.join(self.dest_path, image_name), np.flip(dest_image, axis = -1))


	def __call__(self, srce_path, dest_path):
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

		print('=> => time elapse for model inference:', self.total_elapse / self.total_number)

		print('<= anonimization finishes')


def main(model_path, srce_path, dest_path):
	model = models.create_model(FakeArgs(), model_path)

	with open('res/metadata.json') as file:
		metadata = json.load(file)

	anonymizer = Anonymizer(model, metadata)
	anonymizer(srce_path, dest_path)


if __name__ == '__main__':
	main(sys.argv[1], sys.argv[2], sys.argv[3])
