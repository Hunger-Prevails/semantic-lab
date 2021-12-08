import torch
import segmentation

from segmentation import Init

class FakeArgs:
    def __init__(self):
        self.head = 'deeplabv3'
        self.backbone = 'resnet50'

        self.stride_lift = False

        self.stride = 8
        self.n_classes = 36


def create_model(fargs, model_path):
    print('=> loads checkpoint')
    model_name = fargs.head + '_' + fargs.backbone

    assert hasattr(segmentation, model_name)
    model = getattr(segmentation, model_name)(Init.NONE, fargs.stride, fargs.n_classes, False, fargs.stride_lift)

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
