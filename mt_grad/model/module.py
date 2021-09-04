import torch
import torch.nn as nn

from mt_grad.model.encoder import ResSignalEncoder


class MTParamModel(nn.Module):
    def __init__(self, in_channels: int = 13, hidden_channels: int = 32, out_features: int = 243, use_log: bool = False):
        super(MTParamModel, self).__init__()
        self.rho_encoder = ResSignalEncoder(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            num_heads=2,
            out_channels=hidden_channels
        )

        self.phi_encoder = ResSignalEncoder(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            num_heads=2,
            out_channels=hidden_channels
        )

        self.regression_head = nn.Sequential(
            nn.Linear(hidden_channels * 3, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, out_features),
            nn.ReLU() if not use_log else nn.Identity()
        )

    @staticmethod
    def statistical_pooling(x: torch.Tensor) -> torch.Tensor:
        # (B, C, T) -> (B, C * 3)
        variance = torch.var(x, dim=(2,), unbiased=False)
        std = variance ** 2
        mean = torch.mean(x, dim=(2,))
        return torch.cat([variance, std, mean], dim=1)

    def forward(
            self, rho_yx: torch.Tensor, rho_xy: torch.Tensor, phi_yx: torch.Tensor, phi_xy: torch.Tensor
    ) -> torch.Tensor:
        rho_encoded = self.rho_encoder([rho_yx, rho_xy])
        phi_encoded = self.phi_encoder([phi_yx, phi_xy])
        encoded = rho_encoded + phi_encoded
        encoded = self.statistical_pooling(encoded)
        return self.regression_head(encoded)
