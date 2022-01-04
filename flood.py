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

def statistics(samples):
	center = np.mean(samples, axis = 0)
	covariances = np.matmul((samples - center).T, samples - center)

	return dict(center = center, covariances = covariances)


def rank_by_dist(samples, center, covariances):
	mahalanobis = np.matmul(samples - center, np.linalg.inv(covariances))
	mahalanobis = np.multiply(mahalanobis, samples - center).sum(axis = -1)

	return np.argsort(mahalanobis)


def update(samples, weights, stats):
	ranks = rank_by_dist(samples, stats['center'], stats['covariances'])

	weights[ranks[:samples.shape[0] // 2]] += weights[ranks[:samples.shape[0] // 2]] / 10
	weights[ranks[samples.shape[0] // 2:]] -= weights[ranks[samples.shape[0] // 2:]] / 10

	weights = np.divide(weights, weights.sum())


def ransac(label):
	smask = flood(label)

	samples = np.stack(np.where(smask)).T
	weights = np.ones(samples.shape[0]) / samples.shape[0]

	for i in range(100):
		indices = np.random.choice(samples.shape[0], samples.shape[0] // 2, False)
		stats = statistics(samples[indices])
		update(samples, weights, stats)

	ranks = np.argsort(weights)
	stats = statistics(samples[ranks[samples.shape[0] // 10:]])
	ranks = rank_by_dist(samples, stats['center'], stats['covariances'])

	remains = ranks[:samples.shape[0] - samples.shape[0] // 10]

	new_mask = np.zeros(smask.shape, dtype = bool)
	new_mask[samples[remains, 0], samples[remains, 1]] = True

	return new_mask


def flood(label):
	smask = (label != 0).astype(np.uint8)
	smask = cv2.dilate(smask, None, iterations = 20)
	smask = cv2.erode(smask, None, iterations = 20)

	return smask.astype(bool)


def main(path):
	with open('res/metadata.json') as file:
		metadata = json.load(file)

	annotation = metadata['annotation']['smile_view']

	label = np.load(path)

	plt.figure(figsize = (16, 8))

	ax = plt.subplot(1, 3, 1)
	ax.imshow(utils.to_label_image(label, annotation))

	ax = plt.subplot(1, 3, 2)
	flooded = flood(label)
	ax.imshow(flooded, cmap = 'gray', vmin = 0.0, vmax = 1.0)

	ax = plt.subplot(1, 3, 3)
	dest = ransac(flooded)
	ax.imshow(dest, cmap = 'gray', vmin = 0.0, vmax = 1.0)

	plt.show()

if __name__ == '__main__':
	main(sys.argv[1])