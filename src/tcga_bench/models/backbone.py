"""Full backbone: FeatureTokenizer -> SetEncoder -> representation vector.

Used both for SSL pretraining (via `SSLProjector` on top) and for
downstream classification (via `BayesianLinear` on top). Kept as a single
module so both stages share identical weights end to end.
"""
import torch
import torch.nn as nn

from .tokenizer import FeatureTokenizer
from .set_encoder import SetEncoder


class TabularSetBackbone(nn.Module):
    def __init__(
        self,
        n_features: int,
        dim: int = 64,
        n_heads: int = 4,
        n_inducing: int = 128,
        n_isab_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.tokenizer = FeatureTokenizer(n_features, dim)
        self.encoder = SetEncoder(dim, n_heads, n_inducing, n_isab_layers, dropout)
        self.out_dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.tokenizer(x)
        return self.encoder(tokens)


class PlainMLPBackbone(nn.Module):
    """Ablation backbone: a dense MLP with no feature-identity tokenization
    and no attention -- i.e. an unconstrained, independent weight per input
    coordinate, the opposite inductive bias from TabularSetBackbone. Used
    only to measure what the equivariant/shared-weight design buys us."""

    def __init__(self, n_features: int, dim: int = 64, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 512), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(512, 128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, dim),
        )
        self.out_dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SSLProjector(nn.Module):
    """MLP projection head used only during SSL pretraining (standard
    practice in SimCLR/Barlow-Twins/VICReg: the projector is discarded
    after pretraining, and the backbone representation below it is what
    gets reused downstream)."""

    def __init__(self, in_dim: int, hidden_dim: int = 256, out_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
