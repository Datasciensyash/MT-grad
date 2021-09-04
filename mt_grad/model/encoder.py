import typing as tp

import torch
import torch.nn as nn

from mt_grad.model.blocks import ResEncoder


class ResSignalEncoder(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            hidden_channels: int = 32,
            n_layers: int = 15,
            num_heads: int = 2,
    ):
        super(ResSignalEncoder, self).__init__()

        self.encoder_heads = nn.ModuleList(
            [ResEncoder(
                in_channels=in_channels, hidden_channels=hidden_channels, n_layers=n_layers
            ) for _ in range(num_heads)]
        )

        self.out_projection = nn.Sequential(
            nn.Conv1d(hidden_channels * num_heads, out_channels, kernel_size=(1,)),
            nn.BatchNorm1d(hidden_channels, affine=False),
            nn.LeakyReLU(0.2)
        )

    def forward(self, signals: tp.Tuple[torch.Tensor]) -> torch.Tensor:
        encoded = []
        for i, head in enumerate(self.encoder_heads):
            encoded.append(head(signals[i]))
        encoded = torch.cat(encoded, dim=1)
        encoded = self.out_projection(encoded)
        return encoded
