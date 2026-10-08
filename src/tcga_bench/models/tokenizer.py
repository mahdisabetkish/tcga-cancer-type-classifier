"""
Numeric feature tokenizer for high-dimensional tabular data.

Each scalar feature x_i is mapped to a d-dimensional token

    t_i = phi(x_i) + e_i

where `phi` is a single shared linear map (applied identically to every
feature) and `e_i` is a learned, feature-identity embedding looked up by
index i. Because `phi` is shared across the whole feature axis, the
tokenizer (and anything built from the resulting token sequence using
weights that are themselves shared across tokens, e.g. the ISABEncoder in
`set_encoder.py`) commutes with any permutation of the feature axis, as
long as the identity embeddings are carried along with their values:

    tokenize(x)_{pi(i)} = tokenize_pi(x)_i   for any permutation pi,
    where tokenize_pi reindexes both x and e by pi.

This is the concrete sense in which the architecture is "permutation
equivariant": a plain per-feature weight matrix (as in an ordinary MLP
first layer) has no such symmetry, because each input coordinate owns an
unconstrained, independent weight. Here the only thing that distinguishes
feature i from feature j is the learned embedding e_i, not a separate set
of transformation weights -- a strictly weaker (and much more
parameter-efficient) inductive bias, appropriate when the feature axis is
very wide relative to the number of training rows.
"""
import math

import torch
import torch.nn as nn


class FeatureTokenizer(nn.Module):
    def __init__(self, n_features: int, dim: int):
        super().__init__()
        self.n_features = n_features
        self.dim = dim
        self.value_proj = nn.Linear(1, dim)
        self.feature_embed = nn.Parameter(torch.empty(n_features, dim))
        nn.init.normal_(self.feature_embed, std=1.0 / math.sqrt(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, n_features) -> tokens: (batch, n_features, dim)
        v = self.value_proj(x.unsqueeze(-1))
        return v + self.feature_embed.unsqueeze(0)
