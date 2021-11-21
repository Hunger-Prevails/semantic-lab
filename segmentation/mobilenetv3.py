import warnings
from functools import partial
from typing import Any, Callable, List, Optional, Sequence

import torch
from torch import nn, Tensor
from torch.utils.model_zoo import load_url

from ..ops.misc import SqueezeExcitation as SELayer
from ..ops.misc import ConvNormActivation as CNALayer


__all__ = ['MobileNetV3', 'mobilenet_v3_large', 'mobilenet_v3_small']


model_urls = {
    'mobilenet_v3_large': 'https://download.pytorch.org/models/mobilenet_v3_large-8738ca79.pth',
    'mobilenet_v3_small': 'https://download.pytorch.org/models/mobilenet_v3_small-047dcff4.pth',
}


def round_to(val: int, divisor: int) -> int:
    return torch.mul((val + divisor // 2) // divisor, divisor)


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
        dilation: int,
    ):
        self.input_channels = round_to(input_channels, 8)
        self.kernel = kernel
        self.expanded_channels = round_to(expanded_channels, 8)
        self.out_channels = round_to(out_channels, 8)
        self.use_se = use_se
        self.use_hs = activation == 'HS'
        self.stride = stride
        self.dilation = dilation


class Funnelexit(nn.Module):
    def __init__(
        self,
        config: FunnelexitConfig,
        norm_layer: Callable[..., nn.Module],
    ):
        super().__init__()
        if not (1 <= config.stride <= 2):
            raise ValueError('illegal stride value')

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
    def __init__(
        self,
        block: nn.Module,
        block_configs: List[object],
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()

        if block is None:
            block = Funnelexit

        if norm_layer is None:
            norm_layer = partial(nn.BatchNorm2d, eps = 0.001, momentum = 0.01)

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
        for config in block_configs:
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
        return self.features(x)

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


def _mobilenet_v3(
    arch: str,
    block: nn.Module,
    pretrain: bool = False,
    progress: bool = True,
    **kwargs: Any,
):
    tail_cut = 2 if kwargs.get('tail_cut', False) else 1
    dilation = 2 if kwargs.get('dilation', False) else 1

    bneck_conf = globals()[block.__class__.__name__ + 'Config']

    if arch == 'mobilenet_v3_large':
        block_configs = [
            bneck_conf(16, 3, 16, 16, False, 'RE', 1, 1),
            bneck_conf(16, 3, 64, 24, False, 'RE', 2, 1),
            bneck_conf(24, 3, 72, 24, False, 'RE', 1, 1),
            bneck_conf(24, 5, 72, 40, True, 'RE', 2, 1),
            bneck_conf(40, 5, 120, 40, True, 'RE', 1, 1),
            bneck_conf(40, 5, 120, 40, True, 'RE', 1, 1),
            bneck_conf(40, 3, 240, 80, False, 'HS', 2, 1),
            bneck_conf(80, 3, 200, 80, False, 'HS', 1, 1),
            bneck_conf(80, 3, 184, 80, False, 'HS', 1, 1),
            bneck_conf(80, 3, 184, 80, False, 'HS', 1, 1),
            bneck_conf(80, 3, 480, 112, True, 'HS', 1, 1),
            bneck_conf(112, 3, 672, 112, True, 'HS', 1, 1),
            bneck_conf(112, 5, 672, 160 // tail_cut, True, 'HS', 2, dilation),
            bneck_conf(160 // tail_cut, 5, 960 // tail_cut, 160 // tail_cut, True, 'HS', 1, dilation),
            bneck_conf(160 // tail_cut, 5, 960 // tail_cut, 160 // tail_cut, True, 'HS', 1, dilation),
        ]
    elif arch == 'mobilenet_v3_small':
        block_configs = [
            bneck_conf(16, 3, 16, 16, True, 'RE', 2, 1),
            bneck_conf(16, 3, 72, 24, False, 'RE', 2, 1),
            bneck_conf(24, 3, 88, 24, False, 'RE', 1, 1),
            bneck_conf(24, 5, 96, 40, True, 'HS', 2, 1),
            bneck_conf(40, 5, 240, 40, True, 'HS', 1, 1),
            bneck_conf(40, 5, 240, 40, True, 'HS', 1, 1),
            bneck_conf(40, 5, 120, 48, True, 'HS', 1, 1),
            bneck_conf(48, 5, 144, 48, True, 'HS', 1, 1),
            bneck_conf(48, 5, 288, 96 // tail_cut, True, 'HS', 2, dilation),
            bneck_conf(96 // tail_cut, 5, 576 // tail_cut, 96 // tail_cut, True, 'HS', 1, dilation),
            bneck_conf(96 // tail_cut, 5, 576 // tail_cut, 96 // tail_cut, True, 'HS', 1, dilation),
        ]
    else:
        raise ValueError('Unsupported model type {}'.format(arch))

    model = MobileNetV3(block, block_configs, **kwargs)
    if pretrain:
        if model_urls.get(arch, None) is None:
            raise ValueError('No checkpoint is available for model type {}'.format(arch))
        state_dict = load_state_dict_from_url(model_urls[arch], progress = progress)
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
