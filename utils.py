import torch
import segmentation


class FakeArgs:
    def __init__(self):
        self.head = 'deeplabv3'
        self.backbone = 'resnet50'

        self.aux_loss = False
        self.n_classes = 36


def create_model(fargs, model_path):

    model_name = fargs.head + '_' + fargs.backbone

    assert hasattr(segmentation, model_name)
    model = getattr(segmentation, model_name)(True, True, fargs.n_classes, fargs.aux_loss)

    checkpoint = torch.load(model_path)

    state_keys = set(model.state_dict().keys())
    fetch_keys = set(checkpoint['model'].keys())

    assert not state_keys.difference(fetch_keys)
    assert not fetch_keys.difference(state_keys)

    model.load_state_dict(checkpoint['model'])

    return model.cuda()
