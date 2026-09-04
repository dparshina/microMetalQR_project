import torch
import torch.nn as nn
import torch.nn.functional as F


class SEBlock(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        hidden = max(4, channels // reduction)
        self.fc = nn.Sequential(
            nn.Linear(channels, hidden, bias=False), nn.ReLU(inplace=True),
            nn.Linear(hidden, channels, bias=False), nn.Sigmoid(),
        )

    def forward(self, x):
        batch, channels, _, _ = x.shape
        descriptor = F.adaptive_avg_pool2d(x, 1).view(batch, channels)
        gate = self.fc(descriptor).view(batch, channels, 1, 1)
        return x * gate


class SpatialAttnPre(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)

    def forward(self, x):
        avg_pool = x.mean(dim=1, keepdim=True)
        max_pool, _ = x.max(dim=1, keepdim=True)
        mask = torch.sigmoid(self.conv(torch.cat([avg_pool, max_pool], dim=1)))
        return x * mask


def make_block(in_channels, out_channels, n_conv=2):
    layers = []
    channels = in_channels
    for _ in range(n_conv):
        layers += [nn.Conv2d(channels, out_channels, 3, padding=1),
                   nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True)]
        channels = out_channels
    return nn.Sequential(*layers)


class BlobCNN_Base(nn.Module):
    def __init__(self, in_size=48):
        super().__init__()
        self.b1 = make_block(1, 16)
        self.p1 = nn.MaxPool2d(2)
        self.b2 = make_block(16, 32)
        self.p2 = nn.MaxPool2d(2)
        self.b3 = make_block(32, 64)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Flatten(), nn.Dropout(0.2),
            nn.Linear(64, 32), nn.ReLU(inplace=True),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        x = self.p1(self.b1(x))
        x = self.p2(self.b2(x))
        x = self.gap(self.b3(x))
        return self.head(x).squeeze(-1)


class BlobCNN_SE(nn.Module):
    def __init__(self, in_size=48):
        super().__init__()
        self.b1 = make_block(1, 16); self.se1 = SEBlock(16)
        self.p1 = nn.MaxPool2d(2)
        self.b2 = make_block(16, 32); self.se2 = SEBlock(32)
        self.p2 = nn.MaxPool2d(2)
        self.b3 = make_block(32, 64); self.se3 = SEBlock(64)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Flatten(), nn.Dropout(0.2),
            nn.Linear(64, 32), nn.ReLU(inplace=True),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        x = self.p1(self.se1(self.b1(x)))
        x = self.p2(self.se2(self.b2(x)))
        x = self.gap(self.se3(self.b3(x)))
        return self.head(x).squeeze(-1)


class BlobCNN_SpatialPre(nn.Module):
    def __init__(self, in_size=48):
        super().__init__()
        self.b1a = nn.Sequential(nn.Conv2d(1, 16, 3, padding=1),
                                 nn.BatchNorm2d(16), nn.ReLU(inplace=True))
        self.sap = SpatialAttnPre(kernel_size=7)
        self.b1b = nn.Sequential(nn.Conv2d(16, 16, 3, padding=1),
                                 nn.BatchNorm2d(16), nn.ReLU(inplace=True))
        self.p1 = nn.MaxPool2d(2)
        self.b2 = make_block(16, 32)
        self.p2 = nn.MaxPool2d(2)
        self.b3 = make_block(32, 64)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Flatten(), nn.Dropout(0.2),
            nn.Linear(64, 32), nn.ReLU(inplace=True),
            nn.Linear(32, 1),
        )

    def forward(self, x, return_attn=False):
        x1 = self.b1a(x)
        avg = x1.mean(dim=1, keepdim=True)
        mx, _ = x1.max(dim=1, keepdim=True)
        m = torch.sigmoid(self.sap.conv(torch.cat([avg, mx], dim=1)))
        x1 = x1 * m
        x = self.p1(self.b1b(x1))
        x = self.p2(self.b2(x))
        x = self.gap(self.b3(x))
        logits = self.head(x).squeeze(-1)
        if return_attn:
            return logits, m
        return logits


VARIANTS = {
    'base':    BlobCNN_Base,
    'se':      BlobCNN_SE,
    'spatial': BlobCNN_SpatialPre,
}


def n_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)
