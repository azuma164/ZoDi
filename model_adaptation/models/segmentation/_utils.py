from typing import List

import torch
from torch import nn, Tensor
from torch.nn import functional as F

from models.ssl.byol_head import ContrastiveHead, PredictionHead
from models.ssl.simsiam_head import EncoderHead, PredictorHead


class _SegmentationModel(nn.Module):
    def __init__(self, backbone: nn.Module, classifier: nn.Module) -> None:
        super(_SegmentationModel, self).__init__()
        self.backbone = backbone
        self.classifier = classifier

    def forward(self, x: Tensor) -> Tensor:
        input_shape = x.shape[-2:]

        x1, x2, x3, x4 = self.backbone(x)

        x = self.classifier(x1, x2, x3, x4)
        output = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=False)
        return output

class _SegmentationBYOLModel(nn.Module):
    def __init__(self, backbone: nn.Module, encoder_k: nn.Module, classifier: nn.Module, feat_out_channels: List[int] = [256, 512, 1024, 2048], m: float = 0.99) -> None:
        super(_SegmentationBYOLModel, self).__init__()
        self.m = m

        self.backbone = backbone
        self.encoder_k = encoder_k
        self.classifier = classifier

        out_channel = 128
        self.head_q = ContrastiveHead(
			feat_out_channels=feat_out_channels, out_channel=out_channel)
        self.head_k = ContrastiveHead(
			feat_out_channels=feat_out_channels, out_channel=out_channel)
        self.pred = PredictionHead(heads=4, out_channel=out_channel)

        self._init_encoder_k()

    @torch.no_grad()
    def _init_encoder_k(self):
        for param_k in self.encoder_k.parameters():
            param_k.requires_grad = False  # not update by gradient
        for param_k in self.head_k.parameters():
            param_k.requires_grad = False  # not update by gradient

        try:
            for param_q, param_k in zip(self.backbone.parameters(), self.encoder_k.parameters()):
                param_k.data.copy_(param_q.data)  # initialize
        except:
            for param_q, param_k in zip(self.backbone.module.parameters(), self.encoder_k.parameters()):
                param_k.data.copy_(param_q.data)  # initialize
        for param_q, param_k in zip(self.head_q.parameters(), self.head_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        try:
            for param_q, param_k in zip(self.backbone.parameters(), self.encoder_k.parameters()):
                param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)
        except:
            for param_q, param_k in zip(self.backbone.module.parameters(), self.encoder_k.parameters()):
                param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

        for param_q, param_k in zip(self.head_q.parameters(), self.head_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _momentum_update_key_head(self):
        """
        Momentum update of the key head
        """
        for param_q, param_k in zip(self.head_q.parameters(), self.head_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    def forward(self, x: Tensor, y: Tensor = None, proj: bool = False, dual: bool = False, train_head: bool = False, similarity: bool = False):
        if train_head:
            feats_q = list(self.backbone(x))
            feats_q2 = list(self.backbone(y))
            for i in range(len(feats_q)):
                feats_q[i] = nn.functional.normalize(feats_q[i], dim=1)
                feats_q2[i] = nn.functional.normalize(feats_q2[i], dim=1)
            return feats_q, feats_q2

        x1, x2, x3, x4 = self.backbone(x)

        if y is not None:
            if similarity:
                feat_q = list([x1,x2,x3,x4])
                for i in range(len(feat_q)):
                    feat_q[i] = nn.functional.normalize(feat_q[i], dim=1)
                with torch.no_grad():
                    feat_k = list(self.backbone(y))
                    for i in range(len(feat_k)):
                        feat_k[i] = nn.functional.normalize(feat_k[i], dim=1)
            else:
                feat_q = self.pred(self.head_q([x1,x2,x3,x4]))
                for i in range(len(feat_q)):
                    feat_q[i] = nn.functional.normalize(feat_q[i], dim=1)
                with torch.no_grad():
                    feat_k = self.head_k(self.encoder_k(y))
                    for i in range(len(feat_k)):
                        feat_k[i] = nn.functional.normalize(feat_k[i], dim=1)	
                if dual:
                    feat_q2 = self.pred(self.head_q(self.backbone(y)))
                    for i in range(len(feat_q2)):
                        feat_q2[i] = nn.functional.normalize(feat_q2[i], dim=1)
                    with torch.no_grad():
                        feat_k2 = self.head_k(self.encoder_k(x))
                        for i in range(len(feat_k2)):
                            feat_k2[i] = nn.functional.normalize(feat_k2[i], dim=1)

        input_shape = x.shape[-2:]

        x = self.classifier(x1, x2, x3, x4)
        output = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=False)
        
        if proj:
            return output, x1
        if y is not None:
            if dual:
                return output, feat_q, feat_k, feat_q2, feat_k2
            else:
                return output, feat_q, feat_k
        else:
            return output

class _SegmentationSimSiamModel(nn.Module):
    def __init__(self, backbone: nn.Module, classifier: nn.Module, feat_out_channels: List[int] = [2048]):
        super(_SegmentationSimSiamModel, self).__init__()
        self.backbone = backbone
        self.classifier = classifier

        # out_channel = 2048
        out_channel = 128
        self.encoder = EncoderHead(feat_out_channels, out_channel)
        self.predictor = PredictorHead(feat_out_channels, out_channel)
    
    def forward(self, x: Tensor, y: Tensor = None, proj: bool = False, dual: bool = False, train_head: bool = False, similarity: bool = False):
        if train_head:
            raise NotImplementedError

        x1, x2, x3, x4 = self.backbone(x)

        if y is not None:
            y1, y2, y3, y4 = self.backbone(y)
            if similarity:
                raise NotImplementedError
            else:
                feat_z0 = self.encoder([x4])
                feat_z1 = self.encoder([y4])
                feat_q0 = self.predictor(feat_z0)
                feat_q1 = self.predictor(feat_z1)

                for i in range(len(feat_z0)):
                    feat_z0[i] = nn.functional.normalize(feat_z0[i], dim=1)
                    feat_z0[i].detach()
                for i in range(len(feat_z1)):
                    feat_z1[i] = nn.functional.normalize(feat_z1[i], dim=1)
                    feat_z1[i].detach()

                for i in range(len(feat_q0)):
                    feat_q0[i] = nn.functional.normalize(feat_q0[i], dim=1)
                for i in range(len(feat_q1)):
                    feat_q1[i] = nn.functional.normalize(feat_q1[i], dim=1)

        input_shape = x.shape[-2:]

        x = self.classifier(x1, x2, x3, x4)
        output = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=False)

        if proj:
            return output, x1
        if y is not None:
            if dual:
                return output, feat_q1, feat_z0, feat_q0, feat_z1
            else:
                return output, feat_q1, feat_z0
        else:
            return output

class _SegmentationSimiralityModel(nn.Module):
    def __init__(self, backbone: nn.Module, classifier: nn.Module):
        super(_SegmentationSimiralityModel, self).__init__()
        self.backbone = backbone
        self.classifier = classifier
    
    def forward(self, x: Tensor, y: Tensor = None, proj: bool = False, dual: bool = False, train_head: bool = False, similarity: bool = False):
        if train_head:
            raise NotImplementedError

        x1, x2, x3, x4 = self.backbone(x)

        if y is not None:
            y1, y2, y3, y4 = self.backbone(y)
            if similarity:
                raise NotImplementedError
            else:
                feat_z0 = [x4]
                feat_z1 = [y4]

                for i in range(len(feat_z0)):
                    feat_z0[i] = F.adaptive_avg_pool2d(feat_z0[i], (1, 1))
                    feat_z0[i] = nn.functional.normalize(feat_z0[i], dim=1)
                for i in range(len(feat_z1)):
                    feat_z1[i] = F.adaptive_avg_pool2d(feat_z1[i], (1, 1))
                    feat_z1[i] = nn.functional.normalize(feat_z1[i], dim=1)

        input_shape = x.shape[-2:]

        x = self.classifier(x1, x2, x3, x4)
        output = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=False)

        if proj:
            return output, x1
        if y is not None:
            if dual:
                return output, feat_z1, feat_z0, feat_z0, feat_z1
            else:
                return output, feat_z1, feat_z0
        else:
            return output