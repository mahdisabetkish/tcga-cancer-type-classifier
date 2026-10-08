"""
Stochastic augmentations for tabular self-supervised pretraining, in the
style of SubTab / SAINT / VIME: since there is no rotate/crop/color-jitter
analogue for a feature vector, the standard recipe is corruption of the
feature vector itself.

Two views are drawn independently per sample each training step, so the
SSL objective (see training/vicreg.py) learns representations invariant
to which subset of features was corrupted / how much noise was added --
pushing the model to rely on redundant, distributed signal across many
features rather than memorizing exact values of a few.
"""
import torch


def augment_batch(
    x: torch.Tensor,
    mask_prob: float = 0.15,
    noise_std: float = 0.1,
    scale_jitter: float = 0.05,
) -> torch.Tensor:
    """x: (batch, n_features), already standardized."""
    out = x.clone()

    # feature masking: zero out a random subset of features per sample
    mask = (torch.rand_like(out) < mask_prob)
    out = out.masked_fill(mask, 0.0)

    # additive Gaussian noise
    out = out + noise_std * torch.randn_like(out)

    # per-sample multiplicative scale jitter
    scale = 1.0 + scale_jitter * (2 * torch.rand(out.size(0), 1, device=out.device) - 1)
    out = out * scale

    return out
