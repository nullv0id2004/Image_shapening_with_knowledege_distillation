import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, channels, res_scale=0.1):
        super().__init__()
        self.res_scale = res_scale
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x):
        res = self.conv1(x)
        res = self.relu(res)
        res = self.conv2(res)
        return x + res * self.res_scale  # residual scaling

class SuperResolutionCNN(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, num_channels=64, num_blocks=5, scale_factor=1):
        super().__init__()
        self.entry = nn.Conv2d(in_channels, num_channels, kernel_size=3, padding=1)
        self.res_blocks = nn.Sequential(*[
            ResidualBlock(num_channels, res_scale=0.1) for _ in range(num_blocks)
        ])
        self.exit = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)

        if scale_factor == 1:
            self.upsample = nn.Identity()
        else:
            self.upsample = nn.Sequential(
                nn.Conv2d(num_channels, num_channels * (scale_factor ** 2), kernel_size=3, padding=1),
                nn.PixelShuffle(scale_factor),
                nn.ReLU(inplace=True)
            )

        self.output = nn.Conv2d(num_channels, out_channels, kernel_size=3, padding=1)

        self.apply(self._weights_init)

    def forward(self, x, return_features=False):
        feat = self.entry(x)
        intermediate_feats = []
        res = feat
        for i, block in enumerate(self.res_blocks):
            res = block(res)
            if return_features and i in [0, 2, 4]:
                intermediate_feats.append(res.clone())
        res = self.exit(res)
        up = self.upsample(res + feat)
        out = self.output(up)
        return (out.clamp(0, 1), intermediate_feats) if return_features else out.clamp(0, 1)

    def _weights_init(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
