import math
import json

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


def convert_to_dilation(stride: int):
    std_layer2 = bool(max(3 - math.log2(stride), 0))
    std_layer3 = bool(max(4 - math.log2(stride), 0))
    std_layer4 = bool(max(5 - math.log2(stride), 0))

    return [std_layer2, std_layer3, std_layer4]


def assemble(pretrain: bool, args: object) -> nn.Module:
    backbone = getattr(resnet, args.backbone, getattr(mobilenetv3, args.backbone, None))

    if backbone is None:
        raise NotImplementedError('backbone {} is not supported as of now'.format(args.backbone))

    stride_to_dilation = convert_to_dilation(args.stride)
    backbone = backbone(pretrain = pretrain, stride_to_dilation = stride_to_dilation)

    head_creator = dict(deeplabv3 = DeepLabHead, fcn = FCNHead)
    head_creator = partial(head_creator[args.head], n_classes = args.n_classes)

    aux_head_creator = partial(FCNHead, n_classes = args.n_classes)

    if args.connector is Conn.NONE:
        return Segment(backbone, head_creator)
    elif args.connector is Conn.AUX_NONE:
        return Segment(backbone, head_creator, aux_head_creator)
    elif args.connector is Conn.LIFT:
        return SegmentLift(backbone, head_creator, stride_to_dilation)
    elif args.connector is Conn.AUX_LIFT:
        return SegmentLift(backbone, head_creator, stride_to_dilation, aux_head_creator)
    else:
        return SegmentPyramid(backbone, head_creator, args.n_channels_pyramid)


def assemble_and_load(args: object) -> nn.Module:
    '''
    constructs a segmentation model and optionally loads the coco pre-trains
    '''
    model = assemble(args.pretrain is Init.IMAGENET, args)

    if args.pretrain is Init.COCO:
        load_pretrain(model, args)
    return model


def load_pretrain(model: nn.Module, args: object) -> None:
    arch = args.head + '_' + args.backbone + '_coco'

    with open('res/coco_pretrains.json') as file:
        model_urls = json.load(file)
    assert arch in model_urls

    fetch_dict = load_url(model_urls[arch], progress = True)
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
