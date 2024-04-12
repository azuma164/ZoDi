from torchvision.models.resnet import Bottleneck, BasicBlock, model_urls
from torch.hub import load_state_dict_from_url
import torch

from .resnet import ResNet

# Add custom weights to model urls dict
model_urls['resnet50'] = 'https://download.pytorch.org/models/resnet50-11ad3fa6.pth'
model_urls['resnet101'] = 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth'


def _resnet(arch, block, layers, pretrained, progress, num_classes=1000, invariant=None, **kwargs):
    model = ResNet(block, layers, num_classes=num_classes, **kwargs)

    if pretrained:
        print('Loading {} weights...'.format(arch))
        state_dict = load_state_dict_from_url(model_urls[arch], progress=progress)

        print('Skipping fc layer parameters.')
        del state_dict['fc.weight']
        del state_dict['fc.bias']

        print("invariant: ", invariant)
        if invariant:
            state_dict['conv1.weight'] = torch.sum(state_dict['conv1.weight'], dim=1, keepdim=True)
        # Load weights and print result
        r = model.load_state_dict(state_dict, strict=False)
        print(r)
    return model

def resnet18(pretrained=False, progress=True, **kwargs):
    r"""ResNet-18 model with NConv layer
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    """
    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress, **kwargs)

def resnet34(pretrained=False, progress=True, **kwargs):
    r"""ResNet-18 model with NConv layer
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    """
    return _resnet('resnet34', BasicBlock, [3, 4, 6, 3], pretrained, progress, **kwargs)

def resnet50(pretrained=False, progress=True, **kwargs):
    r"""ResNet-101 model with NConv layer
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    """
    return _resnet('resnet50', Bottleneck, [3, 4, 6, 3], pretrained, progress, **kwargs)

def resnet101(pretrained=False, progress=True, **kwargs):
    r"""ResNet-101 model with NConv layer
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    """
    return _resnet('resnet101', Bottleneck, [3, 4, 23, 3], pretrained, progress, **kwargs)

def resnet151(pretrained=False, progress=True, **kwargs):
    r"""ResNet-101 model with NConv layer
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    """
    return _resnet('resnet151', Bottleneck, [3, 8, 36, 3], pretrained, progress, **kwargs)


if __name__ == '__main__':
    print('Printing ResNet model definition, then exiting.')
    m = resnet101(pretrained=True, num_classes=10)
    print(m)
