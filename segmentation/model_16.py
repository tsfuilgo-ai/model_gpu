from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Sequential):
    """Two 3x3 convolution blocks used by the standard U-Net."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )


class Down(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__(
            nn.MaxPool2d(kernel_size=2, stride=2),
            DoubleConv(in_channels, out_channels),
        )


class Up(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.reduce = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.conv = DoubleConv(2 * out_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = self.reduce(x)
        return self.conv(torch.cat((skip, x), dim=1))


class UNet(nn.Module):
    """A medium-size four-level U-Net for 29-class velocity-model segmentation."""

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 29,
        base_channels: int = 32,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes

        self.encoder1 = DoubleConv(in_channels, base_channels)
        self.encoder2 = Down(base_channels, base_channels * 2)
        self.encoder3 = Down(base_channels * 2, base_channels * 4)
        self.encoder4 = Down(base_channels * 4, base_channels * 8)
        self.bottleneck = Down(base_channels * 8, base_channels * 16)

        self.decoder4 = Up(base_channels * 16, base_channels * 8)
        self.decoder3 = Up(base_channels * 8, base_channels * 4)
        self.decoder2 = Up(base_channels * 4, base_channels * 2)
        self.decoder1 = Up(base_channels * 2, base_channels)
        self.classifier = nn.Conv2d(base_channels, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4 or x.shape[1] != self.in_channels:
            raise ValueError(
                f"x should have shape [B, {self.in_channels}, H, W], got {tuple(x.shape)}"
            )

        skip1 = self.encoder1(x)
        skip2 = self.encoder2(skip1)
        skip3 = self.encoder3(skip2)
        skip4 = self.encoder4(skip3)
        x = self.bottleneck(skip4)

        x = self.decoder4(x, skip4)
        x = self.decoder3(x, skip3)
        x = self.decoder2(x, skip2)
        x = self.decoder1(x, skip1)
        return self.classifier(x)


if __name__ == "__main__":
    model = UNet()
    inputs = torch.randn(2, 1, 128, 66)
    outputs = model(inputs)
    print("parameters:", sum(parameter.numel() for parameter in model.parameters()))
    print("input:", tuple(inputs.shape))
    print("output:", tuple(outputs.shape))
