# Facial Image Anonimization Tool - exoECHO_2DPhotoAnon

### Introduction

This tool anonymizes all images under a given source path with the help of machine learning algorithms.
For a given facial image of a patient, this tool estimates the vertical position of his/her nose within the image.
The part of the image below this position is then culled out and saved to a destination path.
Such an anonymization step ensures that the identity of the patient will not leak.

### Usage

> python path/to/model path/to/source/images path/to/output/images

where

> - path/to/model is the path to a segmentation model.
> - path/to/source/images is the path to images that need to be anonymized.
> - path/to/output/images is the path to which output images are saved.

### Output

> - image_name.png - anonymized images
> - image_name.landmarks.json - facial landmarks of the patient