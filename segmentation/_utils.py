from enum import Enum
from typing import Dict
from typing import Optional
from collections import OrderedDict

from torch import nn, Tensor
from torch.nn import functional as F

from ._misc import FSConvNormActivation as FSCLayer


class Segmentation(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        classifier: nn.Module,
        lifter: Optional[nn.Module] = None,
        aux_classifier: Optional[nn.Module] = None,
        aux_lifter: Optional[nn.Module] = None
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.classifier = classifier
        self.lifter = lifter
        self.aux_classifier = aux_classifier
        self.aux_lifter = aux_lifter


    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        input_shape = x.shape[-2:]
        features = self.backbone(x)

        result = OrderedDict()
        x = features['out']
        if self.lifter is not None:
            x = self.lifter(x)
        x = self.classifier(x)
        x = F.interpolate(x, size = input_shape, mode = 'bilinear', align_corners = False)
        result['out'] = x

        if self.aux_classifier is not None:
            x = features['aux']
            if self.aux_lifter is not None:
                x = self.aux_lifter(x)
            x = self.aux_classifier(x)
            x = F.interpolate(x, size = input_shape, mode = 'bilinear', align_corners = False)
            result['aux'] = x

        return result


class StrideLifter(nn.Sequential):
    def __init__(self, inplanes, size_lifter):
        layers = [FSCLayer(inplanes, inplanes, 4, stride = 2) for i in range(size_lifter)]
        super().__init__(*layers)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


Init = Enum('Init', 'NONE COCO IMAGENET')
