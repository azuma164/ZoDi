import torch
from torch import nn

from .simsiam_head import Encoder, Predictor


class SimSiam(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        dim: int = 2048,
        pred_dim: int = 512,
    ) -> None:

        super().__init__()

        self.dim = dim
        self.pred_dim = pred_dim

        self.backbone = backbone
        self.encoder = Encoder(dim=dim)
        self.predictor = Predictor(dim=dim, pred_dim=pred_dim)

    def forward(self, x0: torch.Tensor, x1: torch.Tensor):
        f0 = self.backbone(x0).flatten(start_dim=1)
        f1 = self.backbone(x1).flatten(start_dim=1)

        z0 = self.encoder(f0)
        z1 = self.encoder(f1)

        p0 = self.predictor(z0)
        p1 = self.predictor(z1)

        return (p0, z0.detach()), (p1, z1.detach())