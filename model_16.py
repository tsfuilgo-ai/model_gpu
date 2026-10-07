from __future__ import annotations

import torch
import torch.nn as nn


class FiLMGenerator(nn.Module):
    """Generate channel-wise FiLM parameters from the water-depth scalar."""

    def __init__(
        self,
        channels: tuple[int, ...] = (4, 8, 16, 32, 64),
        shared_dim: int = 32,
        branch_dim: int = 32,
    ) -> None:
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(1, shared_dim),
            nn.ReLU(inplace=True),
        )
        self.heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(shared_dim, branch_dim),
                    nn.ReLU(inplace=True),
                    nn.Linear(branch_dim, 2 * channel),
                )
                for channel in channels
            ]
        )

        # Start FiLM as an identity transform.
        for head in self.heads:
            nn.init.zeros_(head[-1].weight)
            nn.init.zeros_(head[-1].bias)

    def forward(self, water_depth: torch.Tensor) -> list[tuple[torch.Tensor, torch.Tensor]]:
        if water_depth.ndim == 1:
            water_depth = water_depth.unsqueeze(1)
        if water_depth.ndim != 2 or water_depth.shape[1] != 1:
            raise ValueError(
                f"water_depth should have shape [B] or [B, 1], got {tuple(water_depth.shape)}"
            )

        shared_feature = self.shared(water_depth)
        film_parameters = []
        for head in self.heads:
            delta_gamma, beta = head(shared_feature).chunk(2, dim=1)
            film_parameters.append((delta_gamma, beta))
        return film_parameters


class FiLMConvBlock(nn.Module):
    """Use the Conv2d -> 5D InstanceNorm3d -> FiLM -> ReLU path."""

    def __init__(self, in_channels: int, out_channels: int, data_channels: int = 1) -> None:
        super().__init__()
        self.out_channels = out_channels
        self.data_channels = data_channels
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=5, padding=2)
        self.norm = nn.InstanceNorm3d(data_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(
        self,
        x: torch.Tensor,
        delta_gamma: torch.Tensor,
        beta: torch.Tensor,
        batch_size: int,
        height: int,
        width: int,
    ) -> torch.Tensor:
        x = self.conv(x)
        x = x.reshape(batch_size, self.data_channels, self.out_channels, height, width)
        x = self.norm(x)

        # x is [B, data_channels, conv_channels, H, W]. FiLM modulates
        # each convolution channel and broadcasts over data_channels/H/W.
        delta_gamma = delta_gamma[:, None, :, None, None]
        beta = beta[:, None, :, None, None]
        x = (1.0 + delta_gamma) * x + beta
        x = self.relu(x)
        return x.reshape(batch_size * self.data_channels, self.out_channels, height, width)


class Encoder(nn.Module):
    """Five-layer encoder adapted from tomo_model3.py with FiLM modulation."""

    def __init__(
        self,
        data_channels: int = 1,
        film_shared_dim: int = 32,
        film_branch_dim: int = 32,
    ) -> None:
        super().__init__()
        self.data_channels = data_channels
        channels = (4, 8, 16, 32, 64)
        in_channels = (1,) + channels[:-1]

        self.blocks = nn.ModuleList(
            [
                FiLMConvBlock(c_in, c_out, data_channels=data_channels)
                for c_in, c_out in zip(in_channels, channels)
            ]
        )
        self.film_generator = FiLMGenerator(
            channels=channels,
            shared_dim=film_shared_dim,
            branch_dim=film_branch_dim,
        )

    def forward(self, x: torch.Tensor, water_depth: torch.Tensor) -> torch.Tensor:
        batch_size, data_channels, height, width = x.shape
        if data_channels != self.data_channels:
            raise ValueError(
                f"Encoder expects {self.data_channels} data channel(s), got {data_channels}"
            )

        x = x.reshape(batch_size * data_channels, 1, height, width)
        film_parameters = self.film_generator(water_depth)
        for block, (delta_gamma, beta) in zip(self.blocks, film_parameters):
            x = block(
                x,
                delta_gamma,
                beta,
                batch_size=batch_size,
                height=height,
                width=width,
            )

        # The current network uses data_channels=1, so this is [B, 64, H, W].
        x = x.reshape(batch_size, data_channels * 64, height, width)
        return x


class Generator(nn.Module):
    """Map every seismic trace from the time axis to a latent depth axis."""

    def __init__(
        self,
        num_traces: int = 354,
        time_samples: int = 3000,
        hidden_dims: tuple[int, int, int] = (2048, 1024, 512, 256, 220),
        latent_depth: int = 220,
    ) -> None:
        super().__init__()
        dimensions = (time_samples,) + hidden_dims + (latent_depth,)

        self.layers = nn.ModuleList(
            [nn.Linear(dimensions[i], dimensions[i + 1]) for i in range(len(dimensions) - 1)]
        )
        self.norms = nn.ModuleList([nn.BatchNorm1d(num_traces) for _ in self.layers])
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, channels, num_traces, time_samples = x.shape
        x = x.reshape(batch_size * channels, num_traces, time_samples)

        for linear, norm in zip(self.layers, self.norms):
            x = self.relu(norm(linear(x)))

        return x.transpose(1, 2).contiguous()


def conv_bn_relu(in_channels: int, out_channels: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )


class Decoder(nn.Module):
    """Upsample only the depth axis while preserving the trace axis."""

    def __init__(self) -> None:
        super().__init__()
        self.layer0 = nn.Sequential(
            nn.Upsample(scale_factor=(2, 1), mode="bilinear", align_corners=False),
            conv_bn_relu(64, 64),
            conv_bn_relu(64, 64),
        )
        self.layer1 = nn.Sequential(
            nn.Upsample(scale_factor=(2, 1), mode="bilinear", align_corners=False),
            conv_bn_relu(64, 32),
            conv_bn_relu(32, 32),
        )
        self.layer2 = nn.Sequential(
            conv_bn_relu(32, 16),
            conv_bn_relu(16, 16),
        )
        self.layer3 = nn.Sequential(
            conv_bn_relu(16, 4),
            nn.Conv2d(4, 1, kernel_size=1, bias=False),
            # nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor, batch_size: int) -> torch.Tensor:
        _, depth, num_traces = x.shape
        x = x.reshape(batch_size, 64, depth, num_traces)
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        return self.layer3(x)


class TomoNet(nn.Module):
    """
    End-to-end seismic inversion network.

    Default input:
        x:           [B, 1, 354, 3000]
        water_depth: [B] or [B, 1]

    Default output:
        cropped:     [B, 1, 880, 320]
        uncropped:   [B, 1, 880, 354] when crop_output=False
    """

    def __init__(
        self,
        num_traces: int = 354,
        time_samples: int = 3000,
        generator_dims: tuple[int, int, int] = (2048, 1024, 512),
        latent_depth: int = 220,
        crop_margin: int = 17,
        film_shared_dim: int = 32,
        film_branch_dim: int = 32,
    ) -> None:
        super().__init__()
        self.num_traces = num_traces
        self.time_samples = time_samples
        self.crop_margin = crop_margin

        self.encoder = Encoder(
            film_shared_dim=film_shared_dim,
            film_branch_dim=film_branch_dim,
        )
        self.generator = Generator(
            num_traces=num_traces,
            time_samples=time_samples,
            hidden_dims=generator_dims,
            latent_depth=latent_depth,
        )
        self.decoder = Decoder()

    def forward(
        self,
        x: torch.Tensor,
        water_depth: torch.Tensor,
        crop_output: bool = True,
    ) -> torch.Tensor:
        self._validate_input(x, water_depth)

        batch_size = x.shape[0]
        x = self.encoder(x, water_depth)
        x = self.generator(x)
        x = self.decoder(x, batch_size)

        if crop_output and self.crop_margin > 0:
            x = x[..., self.crop_margin : -self.crop_margin]
        return x

    def _validate_input(self, x: torch.Tensor, water_depth: torch.Tensor) -> None:
        expected_shape = (1, self.num_traces, self.time_samples)
        if x.ndim != 4 or tuple(x.shape[1:]) != expected_shape:
            raise ValueError(
                f"x should have shape [B, {expected_shape[0]}, {expected_shape[1]}, "
                f"{expected_shape[2]}], got {tuple(x.shape)}"
            )
        if water_depth.ndim not in (1, 2) or water_depth.shape[0] != x.shape[0]:
            raise ValueError("water_depth must have the same batch size as x")


if __name__ == "__main__":
    # A small smoke test with the same dimension-changing logic as the full model.
    model = TomoNet(
        num_traces=20,
        time_samples=64,
        generator_dims=(64, 32, 16),
        latent_depth=8,
        crop_margin=2,
    )
    inputs = torch.randn(2, 1, 20, 64)
    depths = torch.tensor([5.0, 10.0])

    full_output = model(inputs, depths, crop_output=False)
    cropped_output = model(inputs, depths)

    print("input:", tuple(inputs.shape))
    print("full output:", tuple(full_output.shape))
    print("cropped output:", tuple(cropped_output.shape))
