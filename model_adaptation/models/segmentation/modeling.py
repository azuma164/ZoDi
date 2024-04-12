from typing import Optional

from models.backbone.resnet import ResNet, resnet50, resnet101
from models.segmentation._deeplabv3 import DeepLabV3, DeepLabV3Similarity, DeepLabV3BYOL, DeepLabV3SimSiam, DeepLabHeadV3Plus, DeepLabHead
from models.segmentation._refinenet import RefineNet, RefineNetHead, RefineNetSimilarity

def _deeplabv3plus_similarity_resnet(
    backbone: ResNet,
    num_classes: int,
) -> DeepLabV3Similarity:
    inplanes = 2048
    low_level_planes = 256
    aspp_dilate = [6,12,18]

    classifier = DeepLabHeadV3Plus(inplanes, low_level_planes, num_classes, aspp_dilate)
    return DeepLabV3Similarity(backbone, classifier)

def deeplabv3plus_similarity_resnet50(
    *,
    backbone_pretrained: bool = True,
    progress: bool = True,
    num_classes: Optional[int] = None,
) -> DeepLabV3BYOL:
    
    if num_classes is None:
        num_classes = 21

    backbone = resnet50(pretrained=backbone_pretrained, progress=progress, return_features=True)
    model = _deeplabv3plus_similarity_resnet(backbone, num_classes)

    return model

def deeplabv3plus_similarity_resnet101(
    *,
    backbone_pretrained: bool = True,
    progress: bool = True,
    num_classes: Optional[int] = None,
) -> DeepLabV3BYOL:
    
    if num_classes is None:
        num_classes = 21

    backbone = resnet101(pretrained=backbone_pretrained, progress=progress, return_features=True)
    model = _deeplabv3plus_similarity_resnet(backbone, num_classes)

    return model

def deeplabv3plus_similarity_ciconv_resnet50(
    *,
    backbone_pretrained: bool = True,
    progress: bool = True,
    num_classes: Optional[int] = None,
) -> DeepLabV3BYOL:
    
    if num_classes is None:
        num_classes = 21

    backbone = resnet50(pretrained=backbone_pretrained, progress=progress, return_features=True, invariant='W')
    model = _deeplabv3plus_similarity_resnet(backbone, num_classes)

    return model

def _deeplabv3plus_byol_resnet(
    backbone: ResNet,
    encoder_k: ResNet,
    num_classes: int,
) -> DeepLabV3BYOL:
    inplanes = 2048
    low_level_planes = 256
    aspp_dilate = [6,12,18]

    classifier = DeepLabHeadV3Plus(inplanes, low_level_planes, num_classes, aspp_dilate)
    return DeepLabV3BYOL(backbone, encoder_k, classifier)

def deeplabv3plus_byol_resnet50(
    *,
    backbone_pretrained: bool = True,
    progress: bool = True,
    num_classes: Optional[int] = None,
) -> DeepLabV3BYOL:
    
    if num_classes is None:
        num_classes = 21

    backbone = resnet50(pretrained=backbone_pretrained, progress=progress, return_features=True)
    encoder_k = resnet50(pretrained=backbone_pretrained, progress=progress, return_features=True)
    model = _deeplabv3plus_byol_resnet(backbone, encoder_k, num_classes)

    return model


def _deeplabv3plus_simsiam_resnet(
    backbone: ResNet,
    num_classes: int,
) -> DeepLabV3BYOL:
    inplanes = 2048
    low_level_planes = 256
    aspp_dilate = [6,12,18]

    classifier = DeepLabHeadV3Plus(inplanes, low_level_planes, num_classes, aspp_dilate)
    return DeepLabV3SimSiam(backbone, classifier)

def deeplabv3plus_simsiam_resnet50(
    *,
    backbone_pretrained: bool = True,
    progress: bool = True,
    num_classes: Optional[int] = None,
) -> DeepLabV3BYOL:
    
    if num_classes is None:
        num_classes = 21

    backbone = resnet50(pretrained=backbone_pretrained, progress=progress, return_features=True)
    model = _deeplabv3plus_simsiam_resnet(backbone, num_classes)

    return model


def _deeplabv3plus_resnet(
    backbone: ResNet,
    num_classes: int,
) -> DeepLabV3:
    inplanes = 2048
    low_level_planes = 256
    aspp_dilate = [6,12,18]

    classifier = DeepLabHeadV3Plus(inplanes, low_level_planes, num_classes, aspp_dilate)
    return DeepLabV3(backbone, classifier)

def deeplabv3plus_resnet50(
    *,
    backbone_pretrained: bool = True,
    progress: bool = True,
    num_classes: Optional[int] = None,
) -> DeepLabV3:
    
    if num_classes is None:
        num_classes = 21

    backbone = resnet50(pretrained=backbone_pretrained, progress=progress, return_features=True)
    model = _deeplabv3plus_resnet(backbone, num_classes)

    return model

def deeplabv3plus_resnet101(
    *,
    backbone_pretrained: bool = True,
    progress: bool = True,
    num_classes: Optional[int] = None,
) -> DeepLabV3:
    
    if num_classes is None:
        num_classes = 21

    backbone = resnet101(pretrained=backbone_pretrained, progress=progress, return_features=True)
    model = _deeplabv3plus_resnet(backbone, num_classes)

    return model


def _deeplabv3_resnet(
    backbone: ResNet,
    num_classes: int,
) -> DeepLabV3:
    classifier = DeepLabHead(2048, num_classes)
    return DeepLabV3(backbone, classifier)

def deeplabv3_resnet50(
    *,
    backbone_pretrained: bool = True,
    progress: bool = True,
    num_classes: Optional[int] = None,
) -> DeepLabV3:
    
    if num_classes is None:
        num_classes = 21

    backbone = resnet50(pretrained=backbone_pretrained, progress=progress, return_features=True)
    model = _deeplabv3_resnet(backbone, num_classes)

    return model


def _refinenet_similarity_resnet(
    backbone: ResNet,
    num_classes: int
) -> RefineNetSimilarity:
    classifier = RefineNetHead(num_classes)
    return RefineNetSimilarity(backbone, classifier)


def refinenet_similarity_resnet50(
    *,
    backbone_pretrained: bool = True,
    progress: bool = True,
    num_classes: Optional[int] = None,
) -> RefineNetSimilarity:
    if num_classes is None:
        num_classes = 21

    backbone = resnet50(pretrained=backbone_pretrained, progress=progress, return_features=True)
    model = _refinenet_similarity_resnet(backbone, num_classes)

    return model

def refinenet_similarity_resnet101(
    *,
    backbone_pretrained: bool = True,
    progress: bool = True,
    num_classes: Optional[int] = None,
) -> RefineNetSimilarity:
    if num_classes is None:
        num_classes = 21

    backbone = resnet101(pretrained=backbone_pretrained, progress=progress, return_features=True)
    model = _refinenet_similarity_resnet(backbone, num_classes)

    return model

def _refinenet_resnet(
    backbone: ResNet,
    num_classes: int
) -> RefineNet:
    classifier = RefineNetHead(num_classes)
    return RefineNet(backbone, classifier)


def refinenet_resnet50(
    *,
    backbone_pretrained: bool = True,
    progress: bool = True,
    num_classes: Optional[int] = None,
) -> RefineNet:
    if num_classes is None:
        num_classes = 21

    backbone = resnet50(pretrained=backbone_pretrained, progress=progress, return_features=True)
    model = _refinenet_resnet(backbone, num_classes)

    return model

def refinenet_resnet101(
    *,
    backbone_pretrained: bool = True,
    progress: bool = True,
    num_classes: Optional[int] = None,
) -> RefineNet:
    if num_classes is None:
        num_classes = 21

    backbone = resnet101(pretrained=backbone_pretrained, progress=progress, return_features=True)
    model = _refinenet_resnet(backbone, num_classes)

    return model
