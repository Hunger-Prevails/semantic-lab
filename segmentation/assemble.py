from torch import nn

from torch.utils.model_zoo import load_url
from . import resnet
from . import mobilenetv3
from .deeplabv3 import DeepLabHead
from .fcn import FCNHead
from ._utils import SegmentationModel


__all__ = [
    'fcn_resnet50',
    'fcn_resnet101',
    'deeplabv3_resnet50',
    'deeplabv3_resnet101',
    'deeplabv3_mobilenet_v3_large',
]


model_urls = {
    'fcn_resnet50_coco': 'https://download.pytorch.org/models/fcn_resnet50_coco-1167a1af.pth',
    'fcn_resnet101_coco': 'https://download.pytorch.org/models/fcn_resnet101_coco-7ecb50ca.pth',
    'deeplabv3_resnet50_coco': 'https://download.pytorch.org/models/deeplabv3_resnet50_coco-cd0a2569.pth',
    'deeplabv3_resnet101_coco': 'https://download.pytorch.org/models/deeplabv3_resnet101_coco-586e9e4e.pth',
    'deeplabv3_mobilenet_v3_large_coco': 'https://download.pytorch.org/models/deeplabv3_mobilenet_v3_large-fc3c493d.pth',
}


def _segm_model(
    name: str, backbone_name: str, num_classes: int, aux_head: bool = False, imagenet: bool = True
) -> nn.Module:
    if 'resnet' in backbone_name:
        backbone = resnet.__dict__[backbone_name](
            pretrain = imagenet,
            stride_to_dilation = [False, True, True]
        )
        out_inplanes = 2048
        aux_inplanes = 1024
    elif 'mobilenet_v3' in backbone_name:
        backbone = mobilenetv3.__dict__[backbone_name](
            pretrain = imagenet,
            stride_to_dilation = [False, False, True, True]
        )
        out_inplanes = 960
        aux_inplanes = 160
    else:
        raise NotImplementedError('backbone {} is not supported as of now'.format(backbone_name))

    aux_classifier = None
    if aux_head:
        aux_classifier = FCNHead(aux_inplanes, num_classes)

    head_creator = dict(deeplabv3 = DeepLabHead, fcn = FCNHead)
    classifier = head_creator[name](out_inplanes, num_classes)

    return SegmentationModel(backbone, classifier, aux_classifier)


def _load_model(
    arch_type: str,
    backbone: str,
    pretrain: bool,
    progress: bool,
    num_classes: int,
    aux_loss: bool = False,
) -> nn.Module:
    '''
    constructs a segmentation model and optionally loads the coco pre-trains
    '''
    model = _segm_model(arch_type, backbone, num_classes, aux_loss, not pretrain)
    if pretrain:
        _load_weights(model, arch_type, backbone, progress)
    return model


def _load_weights(model: nn.Module, arch_type: str, backbone: str, progress: bool) -> None:
    arch = arch_type + '_' + backbone + '_coco'

    assert arch in model_urls

    fetch_dict = load_url(model_urls.get(arch), progress = progress)
    state_dict = model.state_dict()

    fetch_keys = set(fetch_dict.keys())
    state_keys = set(state_dict.keys())

    for key in fetch_keys:
        state_key = key.replace('backbone', 'backbone.features')

        if key in state_dict and fetch_dict.get(key).size() != state_dict.get(key).size():
            print('=> => fetch key [', key, '] deleted due to shape mismatch')
            fetch_dict.pop(key)
        elif key in state_dict:
            pass
        elif state_key in state_dict and fetch_dict.get(key).size() != state_dict.get(state_key).size():
            print('=> => fetch key [', key, '] deleted due to shape mismatch')
            fetch_dict.pop(key)
        elif state_key in state_dict:
            fetch_dict[state_key] = fetch_dict.pop(key)
        else:
            print('=> => fetch key [', key, '] deleted due to redundancy')
            fetch_dict.pop(key)

    for state_key in state_keys.difference(set(fetch_dict.keys())):
        print('=> => state key [', state_key, '] untended')

    state_dict.update(fetch_dict)
    model.load_state_dict(state_dict)


def fcn_resnet50(
    pretrain: bool = False,
    progress: bool = True,
    num_classes: int = 21,
    aux_loss: bool = False,
) -> nn.Module:
    '''Constructs a Fully-Convolutional Network model with a ResNet-50 backbone.

    Args:
        pretrain (bool): If True, returns a model pre-trained on COCO train2017 which
            contains the same classes as Pascal VOC
        progress (bool): If True, displays a progress bar of the download to stderr
        num_classes (int): number of output classes of the model (including the background)
        aux_loss (bool): If True, it uses an auxiliary loss
    '''
    return _load_model('fcn', 'resnet50', pretrain, progress, num_classes, aux_loss)


def fcn_resnet101(
    pretrain: bool = False,
    progress: bool = True,
    num_classes: int = 21,
    aux_loss: bool = False,
) -> nn.Module:
    '''Constructs a Fully-Convolutional Network model with a ResNet-101 backbone.

    Args:
        pretrain (bool): If True, returns a model pre-trained on COCO train2017 which
            contains the same classes as Pascal VOC
        progress (bool): If True, displays a progress bar of the download to stderr
        num_classes (int): number of output classes of the model (including the background)
        aux_loss (bool): If True, it uses an auxiliary loss
    '''
    return _load_model('fcn', 'resnet101', pretrain, progress, num_classes, aux_loss)


def deeplabv3_resnet50(
    pretrain: bool = False,
    progress: bool = True,
    num_classes: int = 21,
    aux_loss: bool = False,
) -> nn.Module:
    '''Constructs a DeepLabV3 model with a ResNet-50 backbone.

    Args:
        pretrain (bool): If True, returns a model pre-trained on COCO train2017 which
            contains the same classes as Pascal VOC
        progress (bool): If True, displays a progress bar of the download to stderr
        num_classes (int): number of output classes of the model (including the background)
        aux_loss (bool): If True, it uses an auxiliary loss
    '''
    return _load_model('deeplabv3', 'resnet50', pretrain, progress, num_classes, aux_loss)


def deeplabv3_resnet101(
    pretrain: bool = False,
    progress: bool = True,
    num_classes: int = 21,
    aux_loss: bool = False,
) -> nn.Module:
    '''Constructs a DeepLabV3 model with a ResNet-101 backbone.

    Args:
        pretrain (bool): If True, returns a model pre-trained on COCO train2017 which
            contains the same classes as Pascal VOC
        progress (bool): If True, displays a progress bar of the download to stderr
        num_classes (int): The number of classes
        aux_loss (bool): If True, include an auxiliary classifier
    '''
    return _load_model('deeplabv3', 'resnet101', pretrain, progress, num_classes, aux_loss)


def deeplabv3_mobilenet_v3_large(
    pretrain: bool = False,
    progress: bool = True,
    num_classes: int = 21,
    aux_loss: bool = False,
) -> nn.Module:
    '''Constructs a DeepLabV3 model with a MobileNetV3-Large backbone.

    Args:
        pretrain (bool): If True, returns a model pre-trained on COCO train2017 which
            contains the same classes as Pascal VOC
        progress (bool): If True, displays a progress bar of the download to stderr
        num_classes (int): number of output classes of the model (including the background)
        aux_loss (bool): If True, it uses an auxiliary loss
    '''
    return _load_model('deeplabv3', 'mobilenet_v3_large', pretrain, progress, num_classes, aux_loss)
