import torch
import numpy as np
import segmentation

from torchvision import transforms

class FakeArgs:
    def __init__(self):
        self.head = 'deeplabv3'
        self.backbone = 'resnet50'

        self.aux_loss = False
        self.n_classes = 36


def create_model(fargs, model_path):
    print('=> loads checkpoint')
    model_name = fargs.head + '_' + fargs.backbone

    assert hasattr(segmentation, model_name)
    model = getattr(segmentation, model_name)(True, True, fargs.n_classes, fargs.aux_loss)

    fetch_dict = torch.load(model_path)['model']
    state_dict = model.state_dict()

    fetch_keys = set(fetch_dict.keys())
    state_keys = set(state_dict.keys())

    for key in fetch_keys.difference(state_keys):
        print('=> => fetch key [', key, '] deleted due to redundancy')
        fetch_dict.pop(key)

    for state_key in state_keys.difference(set(fetch_dict.keys())):
        print('=> => state key [', state_key, '] untended')

    state_dict.update(fetch_dict)
    model.load_state_dict(state_dict)

    return model.cuda()


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


def fetch_rect(label, n_classes):
    h_coords, w_coords = np.where(np.logical_and(0 < label, label < n_classes))

    vert_mass = np.mean(h_coords)

    l_bound, r_bound = np.amin(w_coords), np.amax(w_coords)

    w_range = np.linspace(l_bound, r_bound, num = 9).astype(int)[1:-1]

    shape_marks = [fetch_mark(label, w_coord, n_classes, vert_mass) for w_coord in w_range]
    shape_marks = np.stack(shape_marks).flatten().tolist()

    return np.array(shape_marks + [r_bound - l_bound, 1]), vert_mass


def get_y_coord(jaws_bbox, label, n_classes):
    weights = np.load('res/weights.npy')

    shape_marks, vert_mass = fetch_rect(label, n_classes)

    return vert_mass - np.dot(shape_marks, weights) + jaws_bbox[1]
