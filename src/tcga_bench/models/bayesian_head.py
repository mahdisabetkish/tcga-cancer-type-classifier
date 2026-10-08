"""
Mean-field variational linear layer (Bayes-by-Backprop; Blundell et al.,
2015, "Weight Uncertainty in Neural Networks").

Each weight and bias is a Gaussian q(w) = N(mu, sigma^2) with a learned
mean and a learned log-variance, rather than a single point value. The
training loss adds the KL divergence to a N(0, prior_sigma^2) prior
(scaled by `kl_weight`), so the layer is optimized as a variational
approximation to a Bayesian posterior over its weights, not just
regularized by decay.

At inference, sampling the weights `n_samples` times and averaging the
softmax outputs gives both a prediction and a predictive distribution,
which is what the calibration / uncertainty evaluation in
`eval/calibration.py` consumes -- this is the sense in which the
classifier is "uncertainty-aware" rather than a bare softmax.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class BayesianLinear(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, prior_sigma: float = 1.0):
        super().__init__()
        self.in_dim, self.out_dim = in_dim, out_dim
        self.prior_sigma = prior_sigma

        self.weight_mu = nn.Parameter(torch.empty(out_dim, in_dim))
        self.weight_log_sigma = nn.Parameter(torch.full((out_dim, in_dim), -5.0))
        self.bias_mu = nn.Parameter(torch.zeros(out_dim))
        self.bias_log_sigma = nn.Parameter(torch.full((out_dim,), -5.0))

        nn.init.kaiming_uniform_(self.weight_mu, a=math.sqrt(5))

    def forward(self, x: torch.Tensor, sample: bool = True) -> torch.Tensor:
        if not sample:
            return F.linear(x, self.weight_mu, self.bias_mu)

        w_sigma = torch.exp(self.weight_log_sigma)
        b_sigma = torch.exp(self.bias_log_sigma)
        w = self.weight_mu + w_sigma * torch.randn_like(self.weight_mu)
        b = self.bias_mu + b_sigma * torch.randn_like(self.bias_mu)
        return F.linear(x, w, b)

    def kl_divergence(self) -> torch.Tensor:
        def kl_term(mu, log_sigma):
            sigma2 = torch.exp(2 * log_sigma)
            prior_sigma2 = self.prior_sigma**2
            return 0.5 * torch.sum(
                sigma2 / prior_sigma2
                + mu**2 / prior_sigma2
                - 1.0
                - 2 * log_sigma
                + math.log(prior_sigma2)
            )

        return kl_term(self.weight_mu, self.weight_log_sigma) + kl_term(
            self.bias_mu, self.bias_log_sigma
        )

    @torch.no_grad()
    def predict_distribution(self, x: torch.Tensor, n_samples: int = 30) -> torch.Tensor:
        """Returns softmax probabilities of shape (n_samples, batch, out_dim)."""
        probs = torch.stack([F.softmax(self.forward(x, sample=True), dim=-1) for _ in range(n_samples)])
        return probs
