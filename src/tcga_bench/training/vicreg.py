"""
VICReg loss (Bardes, Ponce & LeCun, 2022, "VICReg: Variance-Invariance-
Covariance Regularization for Self-Supervised Learning").

Unlike SimCLR/contrastive losses, VICReg needs no negative pairs and no
large-batch softmax over the batch, which makes it well suited to a
single-GPU run: it directly regularizes the embedding statistics of two
augmented views z_a, z_b of the same sample:

  - Invariance: MSE(z_a, z_b)            -- the two views should agree
  - Variance:   hinge(sqrt(Var(z)), 1)   -- each embedding dimension must
                                             keep spread (collapse guard)
  - Covariance: off-diagonal(Cov(z))^2   -- decorrelate embedding dims,
                                             pushing them to carry
                                             non-redundant information
"""
import torch
import torch.nn.functional as F


def vicreg_loss(
    z_a: torch.Tensor,
    z_b: torch.Tensor,
    lambda_inv: float = 25.0,
    mu_var: float = 25.0,
    nu_cov: float = 1.0,
    gamma: float = 1.0,
    eps: float = 1e-4,
):
    inv_loss = F.mse_loss(z_a, z_b)

    def variance_loss(z):
        std = torch.sqrt(z.var(dim=0) + eps)
        return torch.mean(F.relu(gamma - std))

    var_loss = variance_loss(z_a) + variance_loss(z_b)

    def covariance_loss(z):
        n, d = z.shape
        z = z - z.mean(dim=0)
        cov = (z.T @ z) / (n - 1)
        off_diag = cov.flatten()[1:].view(d - 1, d + 1)[:, :-1].flatten()
        return (off_diag**2).sum() / d

    cov_loss = covariance_loss(z_a) + covariance_loss(z_b)

    total = lambda_inv * inv_loss + mu_var * var_loss + nu_cov * cov_loss
    return total, {
        "invariance": inv_loss.item(),
        "variance": var_loss.item(),
        "covariance": cov_loss.item(),
    }
