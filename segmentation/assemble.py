import math

from functools import partial

from torch import nn
from torch.utils.model_zoo import load_url

from . import resnet
from . import mobilenetv3
from .fcn import FCNHead
from ._utils import Init
from ._utils import Conn
from ._utils import Segment
from ._utils import SegmentLift
from ._utils import SegmentPyramid
from .deeplabv3 import DeepLabHead


model_urls = {
    'fcn_resnet50_coco': 'https://download.pytorch.org/models/fcn_resnet50_coco-1167a1af.pth',
    'fcn_resnet101_coco': 'https://download.pytorch.org/models/fcn_resnet101_coco-7ecb50ca.pth',
    'deeplabv3_resnet50_coco': 'https://download.pytorch.org/models/deeplabv3_resnet50_coco-cd0a2569.pth',
    'deeplabv3_resnet101_coco': 'https://download.pytorch.org/models/deeplabv3_resnet101_coco-586e9e4e.pth',
    'deeplabv3_mobilenet_v3_large_coco': 'https://download.pytorch.org/models/deeplabv3_mobilenet_v3_large-fc3c493d.pth',
}


def convert_to_dilation(stride: int):
    std_layer2 = bool(max(3 - math.log2(stride), 0))
    std_layer3 = bool(max(4 - math.log2(stride), 0))
    std_layer4 = bool(max(5 - math.log2(stride), 0))

    return [std_layer2, std_layer3, std_layer4]


def assemble(names: dict, pretrain: bool, stride: int, n_classes: int, connector: Conn) -> nn.Module:
    backbone = getattr(resnet, names['backbone'], getattr(mobilenetv3, names['backbone'], None))

    if backbone is None:
        raise NotImplementedError('backbone {} is not supported as of now'.format(names['backbone']))

    stride_to_dilation = convert_to_dilation(stride)
    backbone = backbone(pretrain = pretrain, stride_to_dilation = stride_to_dilation)

    head_creator = dict(deeplabv3 = DeepLabHead, fcn = FCNHead)
    head_creator = partial(head_creator[names['head']], n_classes = n_classes)

    aux_head_creator = partial(FCNHead, n_classes = n_classes)

    if connector is Conn.NONE:
        return Segment(backbone, head_creator)
    elif connector is Conn.AUX_NONE:
        return Segment(backbone, head_creator, aux_head_creator)
    elif connector is Conn.LIFT:
        return SegmentLift(backbone, head_creator, stride_to_dilation)
    elif connector is Conn.AUX_LIFT:
        return SegmentLift(backbone, head_creator, stride_to_dilation, aux_head_creator)
    else:
        return SegmentPyramid(backbone, head_creator)


def _load_model(head: str, backbone: str, pretrain: Init, stride: int, n_classes: int, connector: Conn) -> nn.Module:
    '''
    constructs a segmentation model and optionally loads the coco pre-trains
    '''
    names = dict(head = head, backbone = backbone)
    model = assemble(names, pretrain is Init.IMAGENET, stride, n_classes, connector)

    if pretrain is Init.COCO:
        load_pretrain(model, head, backbone)
    return model


def load_pretrain(model: nn.Module, head: str, backbone: str) -> None:
    arch = head + '_' + backbone + '_coco'

    assert arch in model_urls

    fetch_dict = load_url(model_urls.get(arch), progress = True)
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
    pretrain: Init = Init.NONE,
    stride: int = 16,
    n_classes: int = 21,
    connector: Conn = Conn.NONE
) -> nn.Module:
    '''Constructs a Fully-Convolutional Network model with a ResNet-50 backbone.

    Args:
        n_classes (int): number of output classes of the model (including the background)
    '''
    return _load_model('fcn', 'resnet50', pretrain, stride, n_classes, connector)


def fcn_resnet101(
    pretrain: Init = Init.NONE,
    stride: int = 16,
    n_classes: int = 21,
    connector: Conn = Conn.NONE
) -> nn.Module:
    '''Constructs a Fully-Convolutional Network model with a ResNet-101 backbone.

    Args:
        n_classes (int): number of output classes of the model (including the background)
    '''
    return _load_model('fcn', 'resnet101', pretrain, stride, n_classes, connector)


def deeplabv3_resnet50(
    pretrain: Init = Init.NONE,
    stride: int = 16,
    n_classes: int = 21,
    connector: Conn = Conn.NONE
) -> nn.Module:
    '''Constructs a DeepLabV3 model with a ResNet-50 backbone.

    Args:
        n_classes (int): number of output classes of the model (including the background)
    '''
    return _load_model('deeplabv3', 'resnet50', pretrain, stride, n_classes, connector)


def deeplabv3_resnet101(
    pretrain: Init = Init.NONE,
    stride: int = 16,
    n_classes: int = 21,
    connector: Conn = Conn.NONE
) -> nn.Module:
    '''Constructs a DeepLabV3 model with a ResNet-101 backbone.

    Args:
        n_classes (int): The number of classes
    '''
    return _load_model('deeplabv3', 'resnet101', pretrain, stride, n_classes, connector)


def deeplabv3_mobilenet_v3_large(
    pretrain: Init = Init.NONE,
    stride: int = 16,
    n_classes: int = 21,
    connector: Conn = Conn.NONE
) -> nn.Module:
    '''Constructs a DeepLabV3 model with a MobileNetV3-Large backbone.

    Args:
        n_classes (int): number of output classes of the model (including the background)
    '''
    return _load_model('deeplabv3', 'mobilenet_v3_large', pretrain, stride, n_classes, connector)
