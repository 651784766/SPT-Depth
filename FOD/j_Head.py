import torch.nn as nn
from .j_tools import data_parallel


class Interpolate(nn.Module):
    def __init__(self, scale_factor, mode, align_corners=False):
        super(Interpolate, self).__init__()
        self.interp = nn.functional.interpolate
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, x):
        x = self.interp(
            x,
            scale_factor=self.scale_factor,
            mode=self.mode,
            align_corners=self.align_corners)
        return x


class HeadDepth(nn.Module):
    def __init__(self, hconf, features):
        super(HeadDepth, self).__init__()
        self.hconf = hconf
        self.head = nn.Sequential(
            nn.Conv2d(features, features // 2, kernel_size=3, stride=1, padding=1),
            Interpolate(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(features // 2, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0),
            # nn.ReLU()
            nn.Sigmoid()
        )
        if self.hconf.b_nn_parallel and self.hconf.device_ids is not None:
            self.head = nn.DataParallel(self.head, device_ids=self.hconf.device_ids)
            self.head = self.head.to(self.hconf.device)

    def forward(self, x):
        x = data_parallel(self.head, x, self.hconf)     # x = self.head(x)
        # x = (x - x.min())/(x.max()-x.min() + 1e-15)
        return x


class HeadSeg(nn.Module):
    def __init__(self, hconf, features, nclasses=2):
        super(HeadSeg, self).__init__()
        self.hconf = hconf
        self.head = nn.Sequential(
            nn.Conv2d(features, features // 2, kernel_size=3, stride=1, padding=1),
            Interpolate(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(features // 2, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, nclasses, kernel_size=1, stride=1, padding=0)
        )
        if self.hconf.b_nn_parallel and self.hconf.device_ids is not None:
            self.head = nn.DataParallel(self.head, device_ids=self.hconf.device_ids)
            self.head = self.head.to(self.hconf.device)

    def forward(self, x):
        x = data_parallel(self.head, x, self.hconf)     # x = self.head(x)
        return x
