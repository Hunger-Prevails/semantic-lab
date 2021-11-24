import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import torch
from torchvision import transforms

import json
import utils

def detect_jaws(image, cascade_path):
	faceCascade = cv2.CascadeClassifier(cascade_path)
	bboxes = faceCascade.detectMultiScale(image, minSize = (40, 40))
	bboxes = np.array(bboxes)

	areas = np.multiply(bboxes[:, 2], bboxes[:, 3])

	face_bbox = bboxes[np.argmax(areas)]

	border_t = face_bbox[1] + face_bbox[3] // 2 + face_bbox[3] // 8
	border_b = face_bbox[1] + face_bbox[3]

	border_l = face_bbox[0] + face_bbox[2] // 6
	border_r = face_bbox[0] + face_bbox[2] - face_bbox[2] // 6

	jaws_bbox = np.array([border_l, border_t, border_r - border_l, border_b - border_t])

	return face_bbox, jaws_bbox


def crop_jaws(image, jaws_bbox):
	return image[jaws_bbox[1]:jaws_bbox[1] + jaws_bbox[3], jaws_bbox[0]:jaws_bbox[0] + jaws_bbox[2]].copy()


def detect_teeth(image, model_path, metadata):
	fargs = utils.FakeArgs()
	model = utils.create_model(fargs, model_path)
	model.eval()

	image_transforms = [
		transforms.ToTensor(),
		transforms.Normalize(mean = metadata['mean'], std = metadata['stddvn'])
	]
	image_transforms = transforms.Compose(image_transforms)

	tensor = torch.unsqueeze(image_transforms(image), dim = 0).cuda()

	with torch.no_grad():
		logits = model(tensor)

	return logits['out'].detach().cpu().numpy().argmax(axis = 1).squeeze()


def to_label_image(label, annotation):
	dest_shape = list(label.shape) + [3]
	label_image = np.array([annotation[l] for l in label.flatten()])
	return label_image.reshape(dest_shape).astype(np.uint8)


def main(image_path, cascade_path, model_path, dest_path):
	with open('res/metadata.json') as file:
		metadata = json.load(file)

	annotation = metadata['annotation']['smile_view']

	srce_image = cv2.imread(image_path)
	dest_image = np.flip(srce_image, axis = -1).copy()

	face_bbox, jaws_bbox = detect_jaws(srce_image, cascade_path)

	jaws_image = crop_jaws(dest_image, jaws_bbox)
	mask_image = detect_teeth(jaws_image, model_path, metadata)

	y_coord = utils.get_y_coord(jaws_bbox, mask_image, len(annotation))

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
	ax.imshow(to_label_image(mask_image, annotation))

	ax = plt.subplot(1, 4, 4)
	ax.imshow(dest_image)
	ax.axhline(y = y_coord, color = 'g')

	plt.savefig(dest_path)


if __name__ == '__main__':
	main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
