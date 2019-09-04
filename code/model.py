from efficientnet_pytorch import EfficientNet
from torchvision.models.resnet import ResNet, Bottleneck
from torch.hub import load_state_dict_from_url
import torch.nn as nn
import torch
from config import config

model_urls = {
    'resnext101_32x8d': 'https://download.pytorch.org/models/ig_resnext101_32x8-c38310e5.pth',
    'resnext101_32x16d': 'https://download.pytorch.org/models/ig_resnext101_32x16-c6f796b0.pth',
    'resnext101_32x32d': 'https://download.pytorch.org/models/ig_resnext101_32x32-e4b90b00.pth',
    'resnext101_32x48d': 'https://download.pytorch.org/models/ig_resnext101_32x48-3e41cc8a.pth',
}


# def _resnext(arch, block, layers, pretrained, progress, **kwargs):
#     model = ResNet(block, layers, **kwargs)
#     state_dict = load_state_dict_from_url(model_urls[arch], progress=progress)
#     model.load_state_dict(state_dict)
#     return model

def _resnext(path, block, layers, pretrained, progress, **kwargs):
    model = ResNet(block, layers, **kwargs)
    model.load_state_dict(torch.load(path))
    return model


def resnext101_32x8d_wsl(progress=True, **kwargs):
    """Constructs a ResNeXt-101 32x8 model pre-trained on weakly-supervised data
    and finetuned on ImageNet from Figure 5 in
    `"Exploring the Limits of Weakly Supervised Pretraining" <https://arxiv.org/abs/1805.00932>`_
    Args:
        progress (bool): If True, displays a progress bar of the download to stderr.
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 8
    return _resnext('resnext101_32x8d', Bottleneck, [3, 4, 23, 3], True, progress, **kwargs)


# def resnext101_32x16d_wsl(progress=True, **kwargs):
#     """Constructs a ResNeXt-101 32x16 model pre-trained on weakly-supervised data
#     and finetuned on ImageNet from Figure 5 in
#     `"Exploring the Limits of Weakly Supervised Pretraining" <https://arxiv.org/abs/1805.00932>`_
#     Args:
#         progress (bool): If True, displays a progress bar of the download to stderr.
#     """
#     kwargs['groups'] = 32
#     kwargs['width_per_group'] = 16
#     return _resnext('resnext101_32x16d', Bottleneck, [3, 4, 23, 3], True, progress, **kwargs)

def resnext101_32x16d_wsl(path, progress=True, **kwargs):
    """Constructs a ResNeXt-101 32x16 model pre-trained on weakly-supervised data
    and finetuned on ImageNet from Figure 5 in
    `"Exploring the Limits of Weakly Supervised Pretraining" <https://arxiv.org/abs/1805.00932>`_
    Args:
        progress (bool): If True, displays a progress bar of the download to stderr.
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 16
    return _resnext(path, Bottleneck, [3, 4, 23, 3], True, progress, **kwargs)

def resnext101_32x32d_wsl(progress=True, **kwargs):
    """Constructs a ResNeXt-101 32x32 model pre-trained on weakly-supervised data
    and finetuned on ImageNet from Figure 5 in
    `"Exploring the Limits of Weakly Supervised Pretraining" <https://arxiv.org/abs/1805.00932>`_
    Args:
        progress (bool): If True, displays a progress bar of the download to stderr.
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 32
    return _resnext('resnext101_32x32d', Bottleneck, [3, 4, 23, 3], True, progress, **kwargs)


def resnext101_32x48d_wsl(progress=True, **kwargs):
    """Constructs a ResNeXt-101 32x48 model pre-trained on weakly-supervised data
    and finetuned on ImageNet from Figure 5 in
    `"Exploring the Limits of Weakly Supervised Pretraining" <https://arxiv.org/abs/1805.00932>`_
    Args:
        progress (bool): If True, displays a progress bar of the download to stderr.
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 48
    return _resnext('resnext101_32x48d', Bottleneck, [3, 4, 23, 3], True, progress, **kwargs)


def my_model():
    model = EfficientNet.from_name('efficientnet-b5')
    for param in model.parameters():
        param.requires_grad = False
    in_features = model._fc.in_features
    model._fc = nn.Linear(in_features, config.n_classes)
    # model.load_state_dict(torch.load('weight_best.pt'))
    return model

    # model = resnext101_32x16d_wsl()
    # in_features = model.fc.in_features
    # model.fc = nn.Linear(in_features, configs["train"]["num_classes"])
    # return model

    # model = resnext101_32x16d_wsl(path='../input/ig_resnext101_32x16-c6f796b0.pth')
    # return model

