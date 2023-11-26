import torch
import torch.nn as nn
from .j_tools import data_parallel


class ResidualConvUnit(nn.Module):
    def __init__(self, hconf, features):
        super().__init__()
        self.hconf = hconf
        self.conv1 = nn.Conv2d(
            features, features, kernel_size=3, stride=1, padding=1, bias=True
        )
        self.conv2 = nn.Conv2d(
            features, features, kernel_size=3, stride=1, padding=1, bias=True
        )
        self.relu = nn.ReLU(inplace=True)

        if self.hconf.b_nn_parallel and self.hconf.device_ids is not None:
            self.conv1 = nn.DataParallel(self.conv1, device_ids=self.hconf.device_ids)
            self.conv2 = nn.DataParallel(self.conv2, device_ids=self.hconf.device_ids)
            self.conv1 = self.conv1.to(self.hconf.device)
            self.conv2 = self.conv2.to(self.hconf.device)

    def forward(self, x):
        """Forward pass.
        Args:
            x (tensor): input
        Returns:
            tensor: output
        """
        out = self.relu(x.clone())
        out = data_parallel(self.conv1, out, self.hconf)  # self.conv1(out)
        out = self.relu(out)
        out = data_parallel(self.conv2, out, self.hconf)  # self.conv2(out)
        return out + x


class Fusion(nn.Module):
    def __init__(self, hconf, resample_dim):
        super(Fusion, self).__init__()
        self.hconf = hconf
        self.res_conv1 = ResidualConvUnit(hconf, resample_dim)
        self.res_conv2 = ResidualConvUnit(hconf, resample_dim)
        # self.resample = nn.ConvTranspose2d(resample_dim, resample_dim, kernel_size=2, stride=2, padding=0, bias=True, dilation=1, groups=1)
        if self.hconf.b_nn_parallel and self.hconf.device_ids is not None:
            self.res_conv1 = nn.DataParallel(
                self.res_conv1, device_ids=self.hconf.device_ids
            )
            self.res_conv2 = nn.DataParallel(
                self.res_conv2, device_ids=self.hconf.device_ids
            )
            self.res_conv1 = self.res_conv1.to(self.hconf.device)
            self.res_conv2 = self.res_conv2.to(self.hconf.device)

    def forward(self, x, previous_stage=None):
        if previous_stage is None:
            previous_stage = torch.zeros_like(x)
        output_stage1 = data_parallel(
            self.res_conv1, x, self.hconf
        )  # self.res_conv1(x)
        output_stage1 += previous_stage
        output_stage2 = data_parallel(
            self.res_conv2, output_stage1, self.hconf
        )  # self.res_conv2(output_stage1)
        output_stage2 = nn.functional.interpolate(
            output_stage2, scale_factor=2, mode="bilinear", align_corners=True
        )
        return output_stage2
