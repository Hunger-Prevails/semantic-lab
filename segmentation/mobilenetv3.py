import numpy as np

from functools import partial
from typing import Any, Callable, List, Optional

import torch
from torch import nn, Tensor
from torch.utils.model_zoo import load_url

from .misc import SqueezeExcitation as SELayer
from .misc import ConvNormActivation as CNALayer


__all__ = ['MobileNetV3', 'mobilenet_v3_large', 'mobilenet_v3_small']


model_urls = {
    'mobilenet_v3_large': 'https://download.pytorch.org/models/mobilenet_v3_large-8738ca79.pth',
    'mobilenet_v3_small': 'https://download.pytorch.org/models/mobilenet_v3_small-047dcff4.pth',
}

def round_to(v: float, divisor: int, min_value: Optional[int] = None) -> int:
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)

    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class FunnelexitConfig:
    def __init__(
        self,
        input_channels: int,
        kernel: int,
        expanded_channels: int,
        out_channels: int,
        use_se: bool,
        activation: str,
        stride: int,
    ):
        self.input_channels = round_to(input_channels, 8)
        self.kernel = kernel
        self.expanded_channels = round_to(expanded_channels, 8)
        self.out_channels = round_to(out_channels, 8)
        self.use_se = use_se
        self.use_hs = activation == 'HS'
        self.stride = stride


class Funnelexit(nn.Module):
    def __init__(
        self,
        config: FunnelexitConfig,
        norm_layer: Callable[..., nn.Module],
    ):
        super().__init__()
        print('=> => creates', self.__class__.__name__, 'of stride={:d} and dilation={:d}'.format(config.stride, config.dilation))

        self.use_res_connect = config.stride == 1 and config.input_channels == config.out_channels

        layers: List[nn.Module] = []
        activation_layer = nn.Hardswish if config.use_hs else nn.ReLU

        # expansion
        if config.expanded_channels != config.input_channels:
            layers.append(
                CNALayer(
                    config.input_channels,
                    config.expanded_channels,
                    kernel_size = 1,
                    norm_layer = norm_layer,
                    activation_layer = activation_layer,
                )
            )
        # depthwise
        stride = 1 if config.dilation > 1 else config.stride
        layers.append(
            CNALayer(
                config.expanded_channels,
                config.expanded_channels,
                kernel_size = config.kernel,
                stride = stride,
                dilation = config.dilation,
                groups = config.expanded_channels,
                norm_layer = norm_layer,
                activation_layer = activation_layer,
            )
        )
        if config.use_se:
            squeeze_channels = round_to(config.expanded_channels // 4, 8)
            layers.append(
                SELayer(
                    config.expanded_channels,
                    squeeze_channels,
                    scale_activation = nn.Hardsigmoid
                )
            )
        # projection
        layers.append(
            CNALayer(
                config.expanded_channels,
                config.out_channels,
                kernel_size = 1,
                norm_layer = norm_layer,
                activation_layer = None
            )
        )
        self.block = nn.Sequential(*layers)
        self.out_channels = config.out_channels
        self._is_cn = config.stride > 1

    def forward(self, x: Tensor) -> Tensor:
        result = self.block(x)
        if self.use_res_connect:
            result += x
        return result


class MobileNetV3(nn.Module):
    def __init__(self,
        block: nn.Module,
        block_configs: List[object],
        norm_layer: Callable[..., nn.Module] = None,
        stride_to_dilation: List[object] = None
    ) -> None:
        super().__init__()

        if block is None:
            block = Funnelexit

        if norm_layer is None:
            norm_layer = partial(nn.BatchNorm2d, eps = 0.001, momentum = 0.01)

        if stride_to_dilation is None:
            stride_to_dilation = [False, False, False, False]

        layers: List[nn.Module] = []

        # building first layer
        firstconv_output_channels = block_configs[0].input_channels
        layers.append(
            CNALayer(
                3,
                firstconv_output_channels,
                kernel_size = 3,
                stride = 2,
                norm_layer = norm_layer,
                activation_layer = nn.Hardswish,
            )
        )

        # building inverted residual blocks
        cur_location = 0
        cur_dilation = 1
        for config in block_configs:
            if config.stride > 1:
                if stride_to_dilation[cur_location]:
                    cur_dilation *= config.stride
                    config.stride = 1
                cur_location += 1

            config.dilation = cur_dilation
            layers.append(block(config, norm_layer))

        # building last several layers
        lastconv_input_channels = block_configs[-1].out_channels
        lastconv_output_channels = 6 * lastconv_input_channels
        layers.append(
            CNALayer(
                lastconv_input_channels,
                lastconv_output_channels,
                kernel_size = 1,
                norm_layer = norm_layer,
                activation_layer = nn.Hardswish,
            )
        )
        self.features = nn.Sequential(*layers)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def _forward_impl(self, x: Tensor) -> Tensor:
        for i, child in enumerate(self.features):
            if i < len(self.features) - 3:
                x = child(x)
            elif len(self.features) - 3 < i:
                y = child(y)
            else:
                y = child(x)

        return dict(out = y, aux = x)

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


def _mobilenet_v3(arch: str, block: nn.Module, pretrain: bool = False, progress: bool = True, **kwargs: Any) -> MobileNetV3:
    bneck_conf = globals()[block.__name__ + 'Config']

    if arch == 'mobilenet_v3_large':
        block_configs = [
            bneck_conf(16, 3, 16, 16, False, 'RE', 1),
            bneck_conf(16, 3, 64, 24, False, 'RE', 2),
            bneck_conf(24, 3, 72, 24, False, 'RE', 1),
            bneck_conf(24, 5, 72, 40, True, 'RE', 2),
            bneck_conf(40, 5, 120, 40, True, 'RE', 1),
            bneck_conf(40, 5, 120, 40, True, 'RE', 1),
            bneck_conf(40, 3, 240, 80, False, 'HS', 2),
            bneck_conf(80, 3, 200, 80, False, 'HS', 1),
            bneck_conf(80, 3, 184, 80, False, 'HS', 1),
            bneck_conf(80, 3, 184, 80, False, 'HS', 1),
            bneck_conf(80, 3, 480, 112, True, 'HS', 1),
            bneck_conf(112, 3, 672, 112, True, 'HS', 1),
            bneck_conf(112, 5, 672, 160, True, 'HS', 2),
            bneck_conf(160, 5, 960, 160, True, 'HS', 1),
            bneck_conf(160, 5, 960, 160, True, 'HS', 1),
        ]
    elif arch == 'mobilenet_v3_small':
        block_configs = [
            bneck_conf(16, 3, 16, 16, True, 'RE', 2),
            bneck_conf(16, 3, 72, 24, False, 'RE', 2),
            bneck_conf(24, 3, 88, 24, False, 'RE', 1),
            bneck_conf(24, 5, 96, 40, True, 'HS', 2),
            bneck_conf(40, 5, 240, 40, True, 'HS', 1),
            bneck_conf(40, 5, 240, 40, True, 'HS', 1),
            bneck_conf(40, 5, 120, 48, True, 'HS', 1),
            bneck_conf(48, 5, 144, 48, True, 'HS', 1),
            bneck_conf(48, 5, 288, 96, True, 'HS', 2),
            bneck_conf(96, 5, 576, 96, True, 'HS', 1),
            bneck_conf(96, 5, 576, 96, True, 'HS', 1),
        ]
    else:
        raise ValueError('Unsupported model type {}'.format(arch))

    model = MobileNetV3(block, block_configs, **kwargs)
    if pretrain:
        if model_urls.get(arch, None) is None:
            raise ValueError('No checkpoint is available for model type {}'.format(arch))
        state_dict = load_url(model_urls[arch], progress = progress)
        model.load_state_dict(state_dict)
    return model


def mobilenet_v3_large(pretrain: bool = False, progress: bool = True, **kwargs: Any) -> MobileNetV3:
    '''
    Constructs a large MobileNetV3 architecture from
    `'Searching for MobileNetV3' <https://arxiv.org/abs/1905.02244>`_.

    Args:
        pretrain (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    '''
    return _mobilenet_v3('mobilenet_v3_large', Funnelexit, pretrain, progress, **kwargs)


def mobilenet_v3_small(pretrain: bool = False, progress: bool = True, **kwargs: Any) -> MobileNetV3:
    '''
    Constructs a small MobileNetV3 architecture from
    `'Searching for MobileNetV3' <https://arxiv.org/abs/1905.02244>`_.

    Args:
        pretrain (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    '''
    return _mobilenet_v3('mobilenet_v3_small', Funnelexit, pretrain, progress, **kwargs)
