import math

from torch import nn
from torch.utils.model_zoo import load_url

from . import resnet
from . import mobilenetv3
from .fcn import FCNHead
from ._utils import Init
from ._utils import Segmentation
from ._utils import StrideLifter
from .deeplabv3 import DeepLabHead

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


def std_resnet(stride: int):
    std_layer2 = bool(max(3 - math.log2(stride), 0))
    std_layer3 = bool(max(4 - math.log2(stride), 0))
    std_layer4 = bool(max(5 - math.log2(stride), 0))

    return [std_layer2, std_layer3, std_layer4]


def std_mobilenet(stride: int):
    std_layer1 = bool(max(2 - math.log2(stride), 0))
    std_layer2 = bool(max(3 - math.log2(stride), 0))
    std_layer3 = bool(max(4 - math.log2(stride), 0))
    std_layer4 = bool(max(5 - math.log2(stride), 0))

    return [std_layer1, std_layer2, std_layer3, std_layer4]


def backbone_to_head(
    name: str, backbone_name: str, pretrain: bool, stride: int, n_classes: int, aux_loss: bool = False, stride_lift: bool = False
) -> nn.Module:
    if 'resnet' in backbone_name:
        stride_to_dilation = std_resnet(stride)
        backbone = resnet.__dict__[backbone_name](
            pretrain = pretrain,
            stride_to_dilation = stride_to_dilation
        )
        out_inplanes = 2048
        aux_inplanes = 1024
    elif 'mobilenet_v3' in backbone_name:
        stride_to_dilation = std_mobilenet(stride)
        backbone = mobilenetv3.__dict__[backbone_name](
            pretrain = pretrain,
            stride_to_dilation = stride_to_dilation
        )
        out_inplanes = 960
        aux_inplanes = 160
    else:
        raise NotImplementedError('backbone {} is not supported as of now'.format(backbone_name))

    aux_head = None
    aux_lifter = None
    size_lifter = 1 - stride_to_dilation[-2]
    if aux_loss:
        aux_head = FCNHead(aux_inplanes, n_classes)
        if stride_lift and size_lifter != 0:
            aux_lifter = StrideLifter(aux_inplanes, size_lifter)

    head_creator = dict(deeplabv3 = DeepLabHead, fcn = FCNHead)
    head = head_creator[name](out_inplanes, n_classes)
    lifter = None
    size_lifter = 2 - stride_to_dilation[-2] - stride_to_dilation[-1]
    if stride_lift and size_lifter != 0:
        lifter = StrideLifter(out_inplanes, size_lifter)

    return Segmentation(backbone, head, lifter, aux_head, aux_lifter)


def _load_model(
    arch_type: str,
    backbone: str,
    pretrain: Init,
    stride: int,
    n_classes: int,
    aux_loss: bool,
    stride_lift: bool
) -> nn.Module:
    '''
    constructs a segmentation model and optionally loads the coco pre-trains
    '''
    if pretrain is Init.COCO:
        model = backbone_to_head(arch_type, backbone, False, stride, n_classes, aux_loss, stride_lift)
        load_pretrain(model, arch_type, backbone)
    elif pretrain is Init.IMAGENET:
        model = backbone_to_head(arch_type, backbone, True, stride, n_classes, aux_loss, stride_lift)
    else:
        model = backbone_to_head(arch_type, backbone, False, stride, n_classes, aux_loss, stride_lift)
    return model


def load_pretrain(model: nn.Module, arch_type: str, backbone: str) -> None:
    arch = arch_type + '_' + backbone + '_coco'

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
    pretrain: Init = Init.IMAGENET,
    stride: int = 16,
    n_classes: int = 21,
    aux_loss: bool = False,
    stride_lift: bool = False
) -> nn.Module:
    '''Constructs a Fully-Convolutional Network model with a ResNet-50 backbone.

    Args:
        n_classes (int): number of output classes of the model (including the background)
        aux_loss (bool): If True, it uses an auxiliary loss
    '''
    return _load_model('fcn', 'resnet50', pretrain, stride, n_classes, aux_loss, stride_lift)


def fcn_resnet101(
    pretrain: Init = Init.IMAGENET,
    stride: int = 16,
    n_classes: int = 21,
    aux_loss: bool = False,
    stride_lift: bool = False
) -> nn.Module:
    '''Constructs a Fully-Convolutional Network model with a ResNet-101 backbone.

    Args:
        n_classes (int): number of output classes of the model (including the background)
        aux_loss (bool): If True, it uses an auxiliary loss
    '''
    return _load_model('fcn', 'resnet101', pretrain, stride, n_classes, aux_loss, stride_lift)


def deeplabv3_resnet50(
    pretrain: Init = Init.IMAGENET,
    stride: int = 16,
    n_classes: int = 21,
    aux_loss: bool = False,
    stride_lift: bool = False
) -> nn.Module:
    '''Constructs a DeepLabV3 model with a ResNet-50 backbone.

    Args:
        n_classes (int): number of output classes of the model (including the background)
        aux_loss (bool): If True, it uses an auxiliary loss
    '''
    return _load_model('deeplabv3', 'resnet50', pretrain, stride, n_classes, aux_loss, stride_lift)


def deeplabv3_resnet101(
    pretrain: Init = Init.IMAGENET,
    stride: int = 16,
    n_classes: int = 21,
    aux_loss: bool = False,
    stride_lift: bool = False
) -> nn.Module:
    '''Constructs a DeepLabV3 model with a ResNet-101 backbone.

    Args:
        n_classes (int): The number of classes
        aux_loss (bool): If True, include an auxiliary classifier
    '''
    return _load_model('deeplabv3', 'resnet101', pretrain, stride, n_classes, aux_loss, stride_lift)


def deeplabv3_mobilenet_v3_large(
    pretrain: Init = Init.IMAGENET,
    stride: int = 16,
    n_classes: int = 21,
    aux_loss: bool = False,
    stride_lift: bool = False
) -> nn.Module:
    '''Constructs a DeepLabV3 model with a MobileNetV3-Large backbone.

    Args:
        n_classes (int): number of output classes of the model (including the background)
        aux_loss (bool): If True, it uses an auxiliary loss
    '''
    return _load_model('deeplabv3', 'mobilenet_v3_large', pretrain, stride, n_classes, aux_loss, stride_lift)
