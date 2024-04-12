from torch import nn
from torch.nn import functional as F
from torch.nn import init

class ContrastiveProjMLPV1(nn.Module):
	def __init__(self, in_channel, out_channel, bottle_channel=64):
		super().__init__()
		self.conv1 = nn.Conv2d(in_channel, out_channel, 3, padding=1)
		self.bn1 = nn.BatchNorm1d(out_channel)
		self.relu = nn.ReLU()
		self.fc1 = nn.Linear(out_channel, out_channel)
		self.fc2 = nn.Linear(out_channel, out_channel)
		init.kaiming_normal_(self.fc1.weight)
		init.kaiming_normal_(self.fc2.weight)


	def forward(self, x):
		x = self.conv1(x)
		x = F.adaptive_avg_pool2d(x, (1, 1))
		x = self.fc1(x.view(x.size(0), -1))
		x = self.bn1(x)
		x = self.relu(x)
		x = self.fc2(x)
		return x

class ContrastiveProjMLPV2(nn.Module):
	def __init__(self, in_channel, out_channel, bottle_channel=64):
		super().__init__()
		self.conv1 = nn.Conv2d(in_channel, bottle_channel, 3, padding=1)
		self.bn1 = nn.BatchNorm2d(bottle_channel)
		self.relu = nn.ReLU()
		self.fc1 = nn.Linear(bottle_channel, out_channel)
		init.kaiming_normal_(self.fc1.weight)
		init.kaiming_normal_(self.conv1.weight)
		# init.kaiming_normal_(self.conv2.weight)

	def forward(self, x):
		x = self.conv1(x)
		x = self.relu(self.bn1(x))
		x = F.adaptive_avg_pool2d(x, (1, 1))
		x = self.fc1(x.view(x.size(0), -1))
		return x

class ContrastiveProjMLPV3(nn.Module):
	def __init__(self, in_channel, out_channel, bottle_channel=64):
		super().__init__()
		self.conv1 = nn.Conv2d(in_channel, bottle_channel, 3, padding=1)
		self.bn1 = nn.BatchNorm2d(bottle_channel)
		self.relu = nn.ReLU()
		self.fc1 = nn.Linear(bottle_channel, out_channel)
		self.bn2 = nn.BatchNorm1d(out_channel)
		self.fc2 = nn.Linear(out_channel, out_channel)
		init.kaiming_normal_(self.fc1.weight)
		init.kaiming_normal_(self.conv1.weight)
		init.kaiming_normal_(self.fc2.weight)

	def forward(self, x):
		x = self.conv1(x)
		x = self.relu(self.bn1(x))
		x = F.adaptive_avg_pool2d(x, (1, 1))
		x = self.fc1(x.view(x.size(0), -1))
		x = self.relu(self.bn2(x))
		x = self.fc2(x)
		return x

class ContrastiveMLPConv(nn.Module):
	def __init__(self, in_channel, out_channel, bottle_channel=64):
		super().__init__()
		self.conv1 = nn.Conv2d(in_channel, bottle_channel, 3, padding=1)
		self.relu = nn.ReLU()
		self.conv2 = nn.Conv2d(bottle_channel, bottle_channel, 3, padding=1)
		self.fc = nn.Linear(bottle_channel, out_channel)
		init.kaiming_normal_(self.fc.weight)
		init.kaiming_normal_(self.conv1.weight)
		init.kaiming_normal_(self.conv2.weight)

	def forward(self, x):
		x = self.relu(self.conv2(self.relu(self.conv1(x))))
		x = F.adaptive_avg_pool2d(x, (1, 1))
		x = x.view(x.size(0), x.size(1))
		return self.fc(x)


class ContrastiveHead(nn.Module):
	def __init__(self, feat_out_channels, out_channel=128):
		super().__init__()
		self.single = len(feat_out_channels) == 1
		self.MLPs = []
		for in_channel in feat_out_channels:
			self.MLPs.append(ContrastiveProjMLPV3(in_channel, out_channel))
		self.MLPs = nn.ModuleList(self.MLPs)

	def forward(self, feats, bp=True):
		if self.single:
			return self.MLPs[0](feats)
		outputs = []
		for feat, MLP in zip(feats, self.MLPs):
			if bp:
				outputs.append(MLP(feat))
			else:
				outputs.append(MLP(feat).detach())
		return outputs


class pred_head(nn.Module):
	def __init__(self, out_channel):
		super(pred_head, self).__init__()
		self.in_features = out_channel

		self.fc1 = nn.Linear(out_channel, out_channel)
		self.bn1 = nn.BatchNorm1d(out_channel)
		# self.fc2 = nn.Linear(out_channel, out_channel, bias=False)
		self.bn2 = nn.BatchNorm1d(out_channel)
		self.fc2 = nn.Linear(out_channel, out_channel)

		# init.kaiming_normal_(self.fc1.weight)
		# init.kaiming_normal_(self.fc2.weight)
		# init.eye_(self.fc1.weight)
		# init.eye_(self.fc2.weight)
  
		self.relu = nn.ReLU(inplace=True)

	def forward(self, x):
		# debug

		x = self.fc1(x)
		x = self.bn1(x)

		x = self.relu(x)

		x = self.fc2(x)
		# x = self.bn2(x)

		return x

class PredictionHead(nn.Module):
	def __init__(self, heads=4, out_channel=128):
		super().__init__()
		self.MLPs = []
		for i in range(heads):
			self.MLPs.append(pred_head(out_channel))
		self.MLPs = nn.ModuleList(self.MLPs)

	def forward(self, feats, bp=True):
		outputs = []
		for feat, MLP in zip(feats, self.MLPs):
			if bp:
				outputs.append(MLP(feat))
			else:
				outputs.append(MLP(feat).detach())
		return outputs
