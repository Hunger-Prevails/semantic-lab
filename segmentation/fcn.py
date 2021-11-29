from torch import nn

class FCNHead(nn.Sequential):
    def __init__(self, in_channels: int, n_classes: int) -> None:
        channels = in_channels // 4
        layers = [
            nn.Conv2d(in_channels, channels, 3, padding = 1, bias = False),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(channels, n_classes, 1),
        ]
        super(FCNHead, self).__init__(*layers)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
