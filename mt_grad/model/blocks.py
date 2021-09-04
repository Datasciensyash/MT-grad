import typing as tp

import torch
import torch.nn as nn


class NormConvLayer(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int,
            dilation: int = 1,
            padding: int = 1,
            relu_slope: float = 0.2,
    ):
        super(NormConvLayer, self).__init__()

        self.conv_layer = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(kernel_size,),
            padding=(padding,),
            dilation=(dilation,),
            bias=False
        )

        self.batch_norm = nn.BatchNorm1d(out_channels, affine=False)

        self.activation = nn.LeakyReLU(relu_slope)

    def forward(self, features: torch.Tensor):
        features = self.conv_layer(features)
        features = self.activation(features)
        features = self.batch_norm(features)
        return features


class ResidualBlock(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            size: int = 2,
            kernel_size: int = 3,
            dilation: int = 1,
            padding: int = 1,
            relu_slope: float = 0.2,
    ):
        super(ResidualBlock, self).__init__()
        self.feature_extractors = nn.ModuleList([
            NormConvLayer(in_channels, out_channels, kernel_size, dilation, padding, relu_slope) for _ in range(size)
        ])

    def forward(self, features: torch.Tensor):
        residual_features = features
        for feature_extractor in self.feature_extractors:
            features = feature_extractor(features)
        return features + residual_features


class ResEncoder(nn.Module):
    def __init__(
            self,
            in_channels: int,
            n_layers: int,
            hidden_channels: int,
            size: int = 2,
            use_dilation: bool = True,
            res_pool: tp.Union[bool, int] = False,
    ):
        super(ResEncoder, self).__init__()

        self.in_projection = nn.Conv1d(in_channels, hidden_channels, kernel_size=(1,))

        self.res_blocks = nn.ModuleList()
        self.activation = nn.LeakyReLU(0.2)
        for index in range(n_layers):
            res_block = ResidualBlock(
                in_channels=hidden_channels,
                out_channels=hidden_channels,
                kernel_size=3,
                padding=2 ** (index // 3) if use_dilation else 1,
                dilation=2 ** (index // 3) if use_dilation else 1,
                relu_slope=0.2,
                size=size
            )
            self.res_blocks.append(res_block)

        self.pool = nn.AvgPool2d(res_pool) if res_pool else nn.Identity()

    def forward(self, signal: torch.Tensor):
        hidden = self.in_projection(signal)
        hidden = self.activation(hidden)
        hidden = self.pool(hidden)

        for res_block in self.res_blocks:
            hidden = res_block(hidden)

        return hidden
