import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, mid_channels, 3, 1, padding=1),
            nn.PReLU(mid_channels),
            nn.Conv3d(mid_channels, out_channels, 3, 1, padding=1),
            nn.PReLU(out_channels)
        )

    def forward(self, x):
        return self.double_conv(x)


class ConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels, batch_norm=False):
        super().__init__()
        if batch_norm:
            self.conv = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, 3, 1, padding=1),
                nn.BatchNorm3d(out_channels),
                nn.PReLU(out_channels)
            )
        else:
            self.conv = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, 3, 1, padding=1),
                nn.PReLU(out_channels)
            )

    def forward(self, x):
        return self.conv(x)


class FusionNet(nn.Module):
    """
    using Residual connection
    上一层的输入作为下一层的输入
    """
    def __init__(self):
        super(FusionNet, self).__init__()
        self.conv0 = ConvBlock(32, 16)
        self.conv1 = ConvBlock(48, 16)
        self.conv2 = ConvBlock(64, 16)
        self.conv3 = ConvBlock(80, 16)
        self.conv4 = ConvBlock(96, 16)
        self.conv5 = ConvBlock(16, 16)

    def forward(self, feat1, feat2):
        feat0 = torch.cat((feat1, feat2), dim=1)  # 1,32,32,128,128
        feat1 = self.conv0(feat0)  # 1,16,32,128,128
        feat1 = torch.cat((feat1, feat0), dim=1)  # 1,48,32,128,128
        feat2 = self.conv1(feat1)  # 1,16,32,128,128
        feat2 = torch.cat((feat2, feat1), dim=1)  # 1,64,32,128,128
        feat3 = self.conv2(feat2)  # 1,16,32,128,128
        feat3 = torch.cat((feat3, feat2), dim=1)  # 1,80,32,128,128
        feat4 = self.conv3(feat3)  # 1,16,32,128,128
        feat4 = torch.cat((feat4, feat3), dim=1)  # 1,96,32,128,128
        out = self.conv4(feat4)  # 1,16,32,128,128
        out = self.conv5(out)  # 1,16,32,128,128

        return out




if __name__ == '__main__':
    feat1 = torch.ones((1, 16, 32, 128, 128))
    feat2 = torch.ones((1, 16, 32, 128, 128))
    Net = FusionNet()
    Net.eval()
    y = Net(feat1, feat2)
    print(Net)
    print(y.shape)
