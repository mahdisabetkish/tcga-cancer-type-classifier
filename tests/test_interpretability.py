import torch

from tcga_bench.eval.interpretability import attention_attribution, integrated_gradients
from tcga_bench.models.backbone import TabularSetBackbone
from tcga_bench.models.bayesian_head import BayesianLinear


def test_attention_attribution_sums_to_one():
    # eval() matters: attention dropout would otherwise zero out some
    # weights and break the softmax row-sum-to-1 property being tested
    backbone = TabularSetBackbone(n_features=20, dim=8, n_heads=2, n_inducing=4, n_isab_layers=1).eval()
    x = torch.randn(3, 20)
    attn = attention_attribution(backbone, x)
    assert attn.shape == (3, 20)
    assert torch.allclose(attn.sum(dim=1), torch.ones(3), atol=1e-4)
    assert (attn >= 0).all()


def test_integrated_gradients_shape_and_completeness():
    torch.manual_seed(0)
    backbone = TabularSetBackbone(n_features=12, dim=8, n_heads=2, n_inducing=4, n_isab_layers=1)
    head = BayesianLinear(in_dim=8, out_dim=3)
    backbone.eval()
    head.eval()

    x = torch.randn(1, 12)
    baseline = torch.zeros_like(x)
    target = 1

    attributions = integrated_gradients(backbone, head, x, target, baseline=baseline, n_steps=200)
    assert attributions.shape == (12,)

    with torch.no_grad():
        f_x = head(backbone(x), sample=False)[0, target].item()
        f_baseline = head(backbone(baseline), sample=False)[0, target].item()

    # completeness axiom: sum of attributions ~= f(x) - f(baseline)
    assert abs(attributions.sum().item() - (f_x - f_baseline)) < 0.5
