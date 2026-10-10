"""
Two complementary, model-agnostic-where-possible interpretability tools.
Both operate purely on feature *indices* -- no biological or clinical
label is ever attached to a feature here; see the dashboard's
Methodology tab for the explicit disclaimer this module's outputs are
displayed under.

1. Attention attribution (architecture-specific, free): the PMA pooling
   layer's attention weights are themselves a principled, built-in
   global-importance signal -- they ARE how much weight the model places
   on each feature-token when forming its final representation. No extra
   computation beyond a forward pass with `return_attn=True`.

2. Integrated Gradients (Sundararajan, Taly & Yan, 2017): a standard,
   architecture-agnostic local-attribution method, included because it
   also applies to the PlainMLPBackbone ablation (which has no attention
   to inspect) and because it comes with a checkable correctness property
   (completeness: attributions sum to f(x) - f(baseline)), unlike raw
   gradients or attention weights.
"""
import torch


@torch.no_grad()
def attention_attribution(backbone, x: torch.Tensor) -> torch.Tensor:
    """Mean PMA attention weight per feature token, for a batch of inputs.

    Returns (batch, n_features); requires a backbone exposing
    `return_attn` (TabularSetBackbone), not the PlainMLPBackbone ablation.
    """
    _, attn = backbone(x, return_attn=True)
    return attn  # (batch, n_features), already softmax-normalized, sums to 1 per row


def integrated_gradients(
    backbone,
    head,
    x: torch.Tensor,
    target_class: int,
    baseline: torch.Tensor = None,
    n_steps: int = 50,
) -> torch.Tensor:
    """Integrated Gradients attribution of `head(backbone(x))[target_class]`
    with respect to the input features of a single sample x: (1, n_features).

    Returns (n_features,) attributions. Satisfies (approximately, as
    n_steps -> inf) the completeness axiom:
        attributions.sum() ~= logit(x)[target] - logit(baseline)[target]
    """
    if baseline is None:
        baseline = torch.zeros_like(x)

    alphas = torch.linspace(0, 1, n_steps, device=x.device).view(-1, 1)
    interpolated = baseline + alphas * (x - baseline)  # (n_steps, n_features)
    interpolated.requires_grad_(True)

    rep = backbone(interpolated)
    logits = head(rep, sample=False)
    target_logits = logits[:, target_class]

    grads = torch.autograd.grad(target_logits.sum(), interpolated)[0]  # (n_steps, n_features)
    avg_grads = grads.mean(dim=0)  # trapezoidal-ish Riemann approximation
    attributions = (x.squeeze(0) - baseline.squeeze(0)) * avg_grads
    return attributions.detach()
