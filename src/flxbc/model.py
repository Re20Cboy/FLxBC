from __future__ import annotations


def build_torch_model(input_shape: tuple[int, ...], num_classes: int):
    """Build the lightweight CNN used by the real PyTorch backend."""
    if nn is None:
        raise RuntimeError("PyTorch is not installed. Run `make setup-ml`.")

    if len(input_shape) == 3:
        return Tiny3DMedNet(num_classes=num_classes)
    if len(input_shape) == 2 or len(input_shape) == 3 and input_shape[-1] in (1, 3):
        return Tiny2DMedNet(num_classes=num_classes)
    raise ValueError(f"Unsupported input shape for model: {input_shape}")


try:
    import torch
    from torch import nn
except ImportError:  # pragma: no cover - lets tests run before setup
    torch = None
    nn = None


if nn is not None:

    class DepthwiseSeparableConv3d(nn.Module):
        def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
            super().__init__()
            self.block = nn.Sequential(
                nn.Conv3d(
                    in_channels,
                    in_channels,
                    kernel_size=3,
                    stride=stride,
                    padding=1,
                    groups=in_channels,
                    bias=False,
                ),
                nn.BatchNorm3d(in_channels),
                nn.SiLU(),
                nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm3d(out_channels),
                nn.SiLU(),
            )

        def forward(self, x):
            return self.block(x)

    class Tiny3DMedNet(nn.Module):
        def __init__(self, num_classes: int) -> None:
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv3d(1, 12, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm3d(12),
                nn.SiLU(),
                DepthwiseSeparableConv3d(12, 24, stride=2),
                DepthwiseSeparableConv3d(24, 32, stride=2),
                DepthwiseSeparableConv3d(32, 48, stride=2),
                nn.AdaptiveAvgPool3d(1),
            )
            self.classifier = nn.Linear(48, num_classes)

        def forward(self, x):
            x = self.features(x)
            x = x.flatten(1)
            return self.classifier(x)

    class DepthwiseSeparableConv2d(nn.Module):
        def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
            super().__init__()
            self.block = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    in_channels,
                    kernel_size=3,
                    stride=stride,
                    padding=1,
                    groups=in_channels,
                    bias=False,
                ),
                nn.BatchNorm2d(in_channels),
                nn.SiLU(),
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.SiLU(),
            )

        def forward(self, x):
            return self.block(x)

    class Tiny2DMedNet(nn.Module):
        def __init__(self, num_classes: int) -> None:
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(1, 12, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(12),
                nn.SiLU(),
                DepthwiseSeparableConv2d(12, 24, stride=2),
                DepthwiseSeparableConv2d(24, 32, stride=2),
                nn.AdaptiveAvgPool2d(1),
            )
            self.classifier = nn.Linear(32, num_classes)

        def forward(self, x):
            x = self.features(x)
            x = x.flatten(1)
            return self.classifier(x)

else:

    class Tiny3DMedNet:  # pragma: no cover
        pass

    class Tiny2DMedNet:  # pragma: no cover
        pass
