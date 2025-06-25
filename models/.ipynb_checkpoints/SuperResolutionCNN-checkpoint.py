import torch
import torch.nn as nn
import torch.nn.functional as F
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        )

    def forward(self, x):
        return x + self.block(x)


class SuperResolutionCNN(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, num_channels=64, num_blocks=5, scale_factor=1):
        super(SuperResolutionCNN, self).__init__()

        self.entry = nn.Conv2d(in_channels, num_channels, kernel_size=3, padding=1)

        self.res_blocks = nn.Sequential(
            *[ResidualBlock(num_channels) for _ in range(num_blocks)]
        )

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

        if return_features:
            return out, intermediate_feats
        else:
            return out