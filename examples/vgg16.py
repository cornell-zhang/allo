import torch
import torch.nn as nn
import torch_mlir


cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class VGG(nn.Module):
    def __init__(self, vgg_name):
        super(VGG, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Linear(512, 10)

    def forward(self, x):
        out = self.features(x)
        out = torch.flatten(out, 1)  # out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        index = 0
        for x in cfg:
            if x == 'M':
                layers += []
                # layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            elif cfg[index+1] == 'M':
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1, stride=2, bias=False),
                           nn.ReLU(inplace=True)]
                in_channels = x
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1, bias=False),
                           nn.ReLU(inplace=True)]
                in_channels = x
            index += 1
        layers += [nn.AdaptiveAvgPool2d((1, 1))]
        return nn.Sequential(*layers)


def VGG16():
    return VGG('VGG16')


module = torch_mlir.compile(VGG16(), torch.ones(
    1, 3, 32, 32), output_type=torch_mlir.OutputType.LINALG_ON_TENSORS)

print(module)