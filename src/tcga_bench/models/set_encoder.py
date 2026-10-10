"""
Set-Transformer style encoder (Lee et al., 2019, "Set Transformer: A
Framework for Attention-based Permutation-Invariant Neural Networks").

Two building blocks, both using weights shared across the token axis so
the equivariance property described in `tokenizer.py` is preserved:

  - ISAB (Induced Set Attention Block): instead of O(n^2) self-attention
    over all n feature-tokens, project through a small set of m learned
    inducing points, giving O(n*m) cost. With n in the tens of thousands
    (as it is here) and m in the low hundreds, this is the difference
    between infeasible and cheap.

  - PMA (Pooling by Multihead Attention): a learned, permutation-invariant
    pooling of the (possibly huge) token set down to a fixed number of
    seed vectors, used as the final representation instead of a naive
    mean/sum pool.
"""
import torch
import torch.nn as nn


class MAB(nn.Module):
    """Multihead Attention Block: standard transformer cross-attention
    block (LayerNorm, residual, feed-forward), exposed so it can attend
    from an arbitrary-length query set to an arbitrary-length key/value
    set."""

    def __init__(self, dim: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, n_heads, dropout=dropout, batch_first=True)
        self.ln1 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, 4 * dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(4 * dim, dim)
        )
        self.ln2 = nn.LayerNorm(dim)

    def forward(self, q: torch.Tensor, kv: torch.Tensor, need_weights: bool = False):
        attn_out, attn_w = self.attn(
            q, kv, kv, need_weights=need_weights, average_attn_weights=True
        )
        h = self.ln1(q + attn_out)
        h = self.ln2(h + self.ff(h))
        if need_weights:
            return h, attn_w  # attn_w: (batch, n_queries, n_keys)
        return h


class ISAB(nn.Module):
    def __init__(self, dim: int, n_heads: int, n_inducing: int, dropout: float = 0.1):
        super().__init__()
        self.inducing = nn.Parameter(torch.randn(n_inducing, dim) * (dim**-0.5))
        self.mab1 = MAB(dim, n_heads, dropout)  # inducing <- tokens
        self.mab2 = MAB(dim, n_heads, dropout)  # tokens   <- inducing-contextualized

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = x.size(0)
        inducing = self.inducing.unsqueeze(0).expand(b, -1, -1)
        h = self.mab1(inducing, x)
        return self.mab2(x, h)


class PMA(nn.Module):
    """Pooling by Multihead Attention: k learned seed vectors attend over
    the token set to produce a fixed-size, permutation-invariant summary."""

    def __init__(self, dim: int, n_heads: int, n_seeds: int = 1, dropout: float = 0.1):
        super().__init__()
        self.seeds = nn.Parameter(torch.randn(n_seeds, dim) * (dim**-0.5))
        self.mab = MAB(dim, n_heads, dropout)

    def forward(self, x: torch.Tensor, need_weights: bool = False):
        b = x.size(0)
        seeds = self.seeds.unsqueeze(0).expand(b, -1, -1)
        if need_weights:
            out, attn_w = self.mab(seeds, x, need_weights=True)
            return out, attn_w  # out: (batch, n_seeds, dim), attn_w: (batch, n_seeds, n_tokens)
        return self.mab(seeds, x)  # (batch, n_seeds, dim)


class SetEncoder(nn.Module):
    def __init__(
        self,
        dim: int = 64,
        n_heads: int = 4,
        n_inducing: int = 128,
        n_isab_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.isabs = nn.ModuleList(
            [ISAB(dim, n_heads, n_inducing, dropout) for _ in range(n_isab_layers)]
        )
        self.pma = PMA(dim, n_heads, n_seeds=1, dropout=dropout)

    def forward(self, tokens: torch.Tensor, return_attn: bool = False):
        h = tokens
        for isab in self.isabs:
            h = isab(h)
        if return_attn:
            pooled, attn_w = self.pma(h, need_weights=True)
            return pooled.squeeze(1), attn_w.squeeze(1)  # attn_w: (batch, n_tokens)
        pooled = self.pma(h).squeeze(1)  # (batch, dim)
        return pooled
