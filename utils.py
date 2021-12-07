import torch
import numpy as np

from torchvision import transforms


def detect_teeth(image, model, metadata):
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


def detect_jaws(image, face_cascade):
    bboxes = face_cascade.detectMultiScale(image, minSize = (40, 40))
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


def fetch_mark(label, w_coord, n_classes, vert_mass):
    h_coords = np.where(np.logical_and(0 < label[:, w_coord], label[:, w_coord] < n_classes))[0]
    return [np.amin(h_coords) - vert_mass, np.amax(h_coords) - vert_mass]


def fetch_shape(label, n_classes):
    h_coords, w_coords = np.where(np.logical_and(0 < label, label < n_classes))

    vert_mass = np.mean(h_coords)

    bound_a, bound_b = np.amin(w_coords), np.amax(w_coords)

    w_range = np.linspace(bound_a, bound_b, num = 9).astype(int)[1:-1]

    marks = [fetch_mark(label, w_coord, n_classes, vert_mass) for w_coord in w_range]
    marks = np.stack(marks).flatten().tolist()
    marks = np.array(marks + [bound_b - bound_a, 1])

    return dict(marks = marks, vert_mass = vert_mass)


def get_y_coord(jaws_bbox, label, n_classes):
    weights = np.load('res/weights.npy')

    shape = fetch_shape(label, n_classes)

    return shape['vert_mass'] - np.dot(shape['marks'], weights) + jaws_bbox[1]


def to_label_image(label, annotation):
    dest_shape = list(label.shape) + [3]

    label_image = np.array([annotation[l] for l in label.flatten()])

    return label_image.reshape(dest_shape).astype(np.uint8)
