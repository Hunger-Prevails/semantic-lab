import os
import sys
import cv2
import glob
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import utils
import model


def anonymize(metadata, face_cascade, model, image_file, dest_file):
	print('=> => anonymizes image:', image_file)
	srce_image = cv2.imread(image_file)
	dest_image = np.flip(srce_image, axis = -1).copy()

	try:
		face_bbox, jaws_bbox = utils.detect_jaws(srce_image, face_cascade)
	except:
		print('=> => => detector fails on image:', image_file)
		print('=> => => skips this image')
		return

	jaws_image = utils.crop_jaws(dest_image, jaws_bbox)
	mask_image = utils.detect_teeth(jaws_image, model, metadata)

	annotation = metadata['annotation']['smile_view']
	try:
		y_coord = utils.get_y_coord(jaws_bbox, mask_image, len(annotation))
	except:
		y_coord = jaws_bbox[1]

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

	plt.savefig(dest_file)
	plt.close()


def main(model_path, image_path, dest_path):
	with open('res/metadata.json') as file:
		metadata = json.load(file)

	face_cascade = cv2.CascadeClassifier('res/haarcascade.xml')

	fargs = model.FakeArgs()
	model = model.create_model(fargs, model_path)

	image_files = glob.glob(os.path.join(image_path, '*.png'))
	image_files.sort()

	dest_files = [os.path.join(dest_path, os.path.basename(image_file)) for image_file in image_files]

	print('=> anonimization starts')

	for image_file, dest_file in zip(image_files, dest_files):
		anonymize(metadata, face_cascade, model, image_file, dest_file)

	print('<= anonimization finishes')

if __name__ == '__main__':
	main(sys.argv[1], sys.argv[2], sys.argv[3])
