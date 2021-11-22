import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def detect_jaws(image, cascade_path):
	faceCascade = cv2.CascadeClassifier(cascade_path)
	bboxes = faceCascade.detectMultiScale(image, minSize = (40, 40))
	bboxes = np.array(bboxes)

	border_t = bboxes[:, 1] + bboxes[:, 3] // 2 + bboxes[:, 3] // 8
	border_b = bboxes[:, 1] + bboxes[:, 3]

	border_l = bboxes[:, 0] + bboxes[:, 2] // 6
	border_r = bboxes[:, 0] + bboxes[:, 2] - bboxes[:, 2] // 6

	jaws_bboxes = np.stack([border_l, border_t, border_r - border_l, border_b - border_t]).T

	return bboxes, jaws_bboxes


def crop_jaws(image, jaws_bboxes):
	return [image[bbox[1]:bbox[1] + bbox[3], bbox[0]:bbox[0] + bbox[2]] for bbox in jaws_bboxes]


def detect_teeth(images, model_path):
	pass


def main(image_path, cascade_path, model_path):
	srce_image = cv2.imread(image_path)
	dest_image = np.flip(srce_image, axis = -1).copy()

	face_bboxes, jaws_bboxes = detect_jaws(srce_image, cascade_path)

	jaws_images = crop_jaws(dest_image, jaws_bboxes)
	jaws_maskes = detect_teeth(jaws_images, model_path)

	plt.figure()
	ax = plt.subplot(len(face_bboxes) + 1, 3, 1)
	ax.imshow(dest_image)
	for bbox in face_bboxes:
		rect = patches.Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3], linewidth = 2, edgecolor = 'r', facecolor = 'none')
		ax.add_patch(rect)

	for i_bbox, jaws_image in enumerate(jaws_images):
		ax = plt.subplot(len(face_bboxes) + 1, 3, np.multiply(i_bbox + 1, 3) + 1)
		ax.imshow(jaws_image)
	plt.show()


if __name__ == '__main__':
	main(sys.argv[1], sys.argv[2], sys.argv[3])
