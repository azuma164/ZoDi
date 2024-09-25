import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import init

class Encoder(nn.Module): # original: out: 128
	def __init__(self, in_channel: int, out_channel: int = 2048, bottle_channel: int = 64) -> None:
		super().__init__()
		self.conv1 = nn.Conv2d(in_channel, bottle_channel, 3, padding=1)
		self.bn1 = nn.BatchNorm2d(bottle_channel)
		self.relu = nn.ReLU(inplace=True)
		self.fc1 = nn.Linear(bottle_channel, bottle_channel, bias=False)
		self.bn2 = nn.BatchNorm1d(bottle_channel)
		self.fc2 = nn.Linear(bottle_channel, out_channel, bias=False)
		self.bn3 = nn.BatchNorm1d(out_channel, affine=False)

	def forward(self, x) -> torch.Tensor:
		x = self.conv1(x)
		# x = self.relu(self.bn1(x))
		x = F.adaptive_avg_pool2d(x, (1, 1))
		x = self.fc1(x.view(x.size(0), -1))
		x = self.bn2(x)
		x = self.relu(x)
		x = self.fc2(x)
		x = self.bn3(x)
		# x = self.relu(x)
		# x = self.bn3(x)

		return x
    
class Predictor(nn.Module): # original: 128, 128
	def __init__(self, dim: int = 2048, pred_dim: int = 128) -> None:
		super().__init__()

		self.layer = nn.Sequential(
			nn.Linear(dim, pred_dim, bias=False),
			nn.BatchNorm1d(pred_dim),
			nn.ReLU(inplace=True),  # hidden layer
			nn.Linear(pred_dim, dim),  # output layer
        )

	def forward(self, x) -> torch.Tensor:
		return self.layer(x)

class EncoderHead(nn.Module):
	def __init__(self, feat_out_channels, out_channel=2048):
		super().__init__()
		self.single = len(feat_out_channels) == 1
		self.MLPs = []
		for in_channel in feat_out_channels:
			self.MLPs.append(Encoder(in_channel, out_channel))
		self.MLPs = nn.ModuleList(self.MLPs)

	def forward(self, feats, bp=True):
		outputs = []
		for feat, MLP in zip(feats, self.MLPs):
			if bp:
				outputs.append(MLP(feat))
			else:
				outputs.append(MLP(feat).detach())
		return outputs

class PredictorHead(nn.Module):
	def __init__(self, feat_out_channels, out_channel=2048):
		super().__init__()
		self.single = len(feat_out_channels) == 1
		self.MLPs = []
		for in_channel in feat_out_channels:
			self.MLPs.append(Predictor(out_channel))
		self.MLPs = nn.ModuleList(self.MLPs)

	def forward(self, feats, bp=True):
		outputs = []
		for feat, MLP in zip(feats, self.MLPs):
			if bp:
				outputs.append(MLP(feat))
			else:
				outputs.append(MLP(feat).detach())
		return outputs