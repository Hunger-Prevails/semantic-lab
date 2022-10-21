import numpy as np

from exceptions import BoxException

from torch.nn import functional as F

def detect_teeth(image, model, metadata):
    import torch
    from torchvision import transforms
    model.eval()

    image_transforms = [
        transforms.ToTensor(),
        transforms.Normalize(mean = metadata['mean'], std = metadata['stddvn'])
    ]
    image_transforms = transforms.Compose(image_transforms)

    tensor = torch.unsqueeze(image_transforms(image), dim = 0).cuda()
    tensor_shape = tensor.shape[2:]
    tensor = F.interpolate(tensor, size = [320, 480], mode = 'bilinear', align_corners = False)

    with torch.no_grad():
        logits = model(tensor)
        logits = F.interpolate(logits['layer4'].detach(), size = tensor_shape)

    return logits.cpu().numpy().argmax(axis = 1).squeeze()


def landmarks_to_bbox(face_marks):
    chin_marks = np.stack(face_marks['chin'])
    nose_marks = np.stack(face_marks['nose_tip'])
    lips_marks = np.stack(face_marks['bottom_lip'] + face_marks['top_lip'])

    border_t = nose_marks[:, 1].mean()
    border_b = chin_marks[4:-4, 1].mean()

    lips_l_margin = np.amin(lips_marks[:, 0])
    lips_r_margin = np.amax(lips_marks[:, 0])

    lips_center = (lips_l_margin + lips_r_margin) // 2
    jaws_bbox_w = (lips_r_margin - lips_l_margin) // (3 / 4)

    border_l = lips_center - jaws_bbox_w // 2
    border_r = lips_center + jaws_bbox_w // 2

    return np.array([border_l, border_t, border_r - border_l, border_b - border_t]).astype(int)


def detect_jaws(image):
    import face_recognition
    locations = face_recognition.face_locations(image)
    landmarks = face_recognition.face_landmarks(image, locations)
    locations = np.array(locations)

    bboxes = np.stack([locations[:, 3], locations[:, 0], locations[:, 1] - locations[:, 3], locations[:, 2] - locations[:, 0]]).T

    box_areas = np.multiply(bboxes[:, 2], bboxes[:, 3])
    face_indx = np.argmax(box_areas)

    face_bbox = bboxes[face_indx]
    face_marks = landmarks[face_indx]

    jaws_bbox = landmarks_to_bbox(face_marks)

    return face_bbox, jaws_bbox, face_marks


def crop_jaws(image, jaws_bbox):
    return image[jaws_bbox[1]:jaws_bbox[1] + jaws_bbox[3], jaws_bbox[0]:jaws_bbox[0] + jaws_bbox[2]].copy()


def fetch_mark(smask, w_coord, vert_mass):
    h_coords = np.where(smask[:, w_coord])[0]
    return [np.amin(h_coords) - vert_mass, np.amax(h_coords) - vert_mass]


def fetch_shape(smask):
    h_coords, w_coords = np.where(smask)

    vert_mass = np.mean(h_coords)

    bound_a, bound_b = np.amin(w_coords), np.amax(w_coords)

    w_range = np.linspace(bound_a, bound_b, num = 9).astype(int)[1:-1]

    marks = [fetch_mark(smask, w_coord, vert_mass) for w_coord in w_range]
    marks = np.stack(marks).flatten().tolist()
    marks = np.array(marks + [bound_b - bound_a, 1])

    return dict(marks = marks, vert_mass = vert_mass)


def get_y_coord(jaws_bbox, smask):
    weights = np.load('res/weights.npy')

    shape = fetch_shape(smask)

    return shape['vert_mass'] - np.dot(shape['marks'], weights) + jaws_bbox[1]


def to_label_image(label, annotation):
    dest_shape = list(label.shape) + [3]

    label_image = np.array([annotation[l] for l in label.flatten()])

    return label_image.reshape(dest_shape).astype(np.uint8)
