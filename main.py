import os
import sys
import cv2
import glob
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import utils
import models

from models import FakeArgs

np.seterr(all = 'raise')

class Anonymizer(object):

	def __init__(self, model, metadata, face_cascade):

		self.model = model
		self.metadata = metadata
		self.face_cascade = face_cascade


	def anonymize(self, image_path):
		image_name = os.path.basename(image_path)

		if os.path.exists(os.path.join(self.dest_path, image_name)):
			return
		print('=> => anonymizes image:', image_name)

		srce_image = cv2.imread(image_path)
		dest_image = np.flip(srce_image, axis = -1).copy()

		try:
			face_bbox, jaws_bbox = utils.detect_jaws(srce_image, self.face_cascade)
		except:
			print('=> => => detect fails on', image_path)
			print('=> => => copies image to', self.fail_path['detect'])

			if not os.path.exists(self.fail_path['detect']):
				os.mkdir(self.fail_path['detect'])
			cv2.imwrite(os.path.join(self.fail_path['detect'], image_name), srce_image)
			return

		jaws_image = utils.crop_jaws(dest_image, jaws_bbox)
		mask_image = utils.detect_teeth(jaws_image, self.model, self.metadata)

		annotation = self.metadata['annotation']['smile_view']
		try:
			y_coord = utils.get_y_coord(jaws_bbox, mask_image, len(annotation))
		except:
			print('=> => => smodel fails on', image_path)
			print('=> => => copies image to', self.fail_path['smodel'])

			if not os.path.exists(self.fail_path['smodel']):
				os.mkdir(self.fail_path['smodel'])
			cv2.imwrite(os.path.join(self.fail_path['smodel'], image_name), srce_image)
			return

		plt.figure(figsize = (16, 8))

		ax = plt.subplot(1, 4, 1)
		ax.imshow(dest_image)
		rect = patches.Rectangle((face_bbox[0], face_bbox[1]), face_bbox[2], face_bbox[3], linewidth = 2, edgecolor = 'r', facecolor = 'none')
		ax.add_patch(rect)
		rect = patches.Rectangle((jaws_bbox[0], jaws_bbox[1]), jaws_bbox[2], jaws_bbox[3], linewidth = 2, edgecolor = 'b', facecolor = 'none')
		ax.add_patch(rect)

		ax = plt.subplot(1, 4, 2)
		ax.imshow(jaws_image)

		ax = plt.subplot(1, 4, 3)
		ax.imshow(utils.to_label_image(mask_image, annotation))

		ax = plt.subplot(1, 4, 4)
		ax.imshow(dest_image)
		ax.axhline(y = y_coord, color = 'g')

		plt.savefig(os.path.join(self.dest_path, image_name))
		plt.close()


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

	face_cascade = cv2.CascadeClassifier('res/haarcascade.xml')

	anonymizer = Anonymizer(model, metadata, face_cascade)
	anonymizer.run(srce_path, dest_path)


if __name__ == '__main__':
	main(sys.argv[1], sys.argv[2], sys.argv[3])
