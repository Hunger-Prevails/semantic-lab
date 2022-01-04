from enum import Enum
from typing import List
from typing import Dict
from typing import Optional
from typing import Callable
from collections import OrderedDict

from torch import nn, Tensor
from torch.nn import functional as F

from ._misc import ConvNormActivation as CNALayer
from ._misc import FSConvNormActivation as FSCLayer


class StrideLifter(nn.Sequential):
    def __init__(self, inplanes, size_lifter):
        layers = []
        for i in range(size_lifter):
            layers.append(FSCLayer(inplanes, inplanes // 2, 4, stride = 2))
            inplanes = inplanes // 2
        super().__init__(*layers)
        self.outplanes = inplanes

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


class Segment(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        classifier: Callable[..., nn.Module],
        aux_classifier: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.classifier = classifier(backbone.outplanes['layer4'])
        self.aux_classifier = None
        if aux_classifier:
            self.aux_classifier = aux_classifier(backbone.outplanes['layer3'])

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        input_shape = x.shape[2:]
        features = self.backbone(x)

        result = OrderedDict()
        x = features['layer4']
        x = self.classifier(x)
        x = F.interpolate(x, size = input_shape, mode = 'bilinear', align_corners = False)
        result['layer4'] = x

        if self.aux_classifier:
            x = features['layer3']
            x = self.aux_classifier(x)
            x = F.interpolate(x, size = input_shape, mode = 'bilinear', align_corners = False)
            result['layer3'] = x

        return result


class SegmentLift(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        classifier: Callable[..., nn.Module],
        stride_to_dilation: List[bool],
        aux_classifier: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.lifter = StrideLifter(backbone.outplanes['layer4'], 2 - stride_to_dilation[-2] - stride_to_dilation[-1])
        self.classifier = classifier(self.lifter.outplanes)

        self.aux_classifier = None
        if aux_classifier:
            self.aux_lifter = StrideLifter(backbone.outplanes['layer3'], 1 - stride_to_dilation[-2])
            self.aux_classifier = aux_classifier(self.aux_lifter.outplanes)

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        input_shape = x.shape[2:]
        features = self.backbone(x)

        result = OrderedDict()
        x = features['layer4']
        x = self.lifter(x)
        x = self.classifier(x)
        x = F.interpolate(x, size = input_shape, mode = 'bilinear', align_corners = False)
        result['layer4'] = x

        if self.aux_classifier:
            x = features['layer3']
            x = self.aux_lifter(x)
            x = self.aux_classifier(x)
            x = F.interpolate(x, size = input_shape, mode = 'bilinear', align_corners = False)
            result['layer3'] = x

        return result


class SegmentPyramid(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        classifier: Callable[..., nn.Module],
        n_channels: int
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.classifier = classifier(n_channels)

        self.layer1_1x1 = CNALayer(backbone.outplanes['layer1'], n_channels, 1, stride = 2)
        self.layer2_1x1 = CNALayer(backbone.outplanes['layer2'], n_channels, 1, stride = 1)
        self.layer3_1x1 = CNALayer(backbone.outplanes['layer3'], n_channels, 1, stride = 1)
        self.layer4_1x1 = CNALayer(backbone.outplanes['layer4'], n_channels, 1, stride = 1)

        self.conv = CNALayer(n_channels, n_channels, 3)

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        input_shape = x.shape[2:]
        features = self.backbone(x)

        x1 = features['layer1']
        x2 = features['layer2']
        x3 = features['layer3']
        x4 = features['layer4']

        x1 = self.layer1_1x1(x1)
        x2 = self.layer2_1x1(x2)
        x3 = self.layer3_1x1(x3)
        x4 = self.layer4_1x1(x4)

        x = x4
        x = F.interpolate(x, size = x3.shape[2:], mode = 'bilinear', align_corners = False)
        x += x3
        x = F.interpolate(x, size = x2.shape[2:], mode = 'bilinear', align_corners = False)
        x += x2
        x += x1

        x = self.conv(x)
        x = self.classifier(x)
        x = F.interpolate(x, size = input_shape, mode = 'bilinear', align_corners = False)

        return dict(layer4 = x)


Init = Enum('Init', 'NONE COCO IMAGENET')
Conn = Enum('Conn', 'NONE AUX_NONE LIFT AUX_LIFT PYRAMID')
