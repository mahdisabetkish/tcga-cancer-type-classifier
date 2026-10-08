import torch

from tcga_bench.models.backbone import PlainMLPBackbone, SSLProjector, TabularSetBackbone
from tcga_bench.models.bayesian_head import BayesianLinear
from tcga_bench.models.tokenizer import FeatureTokenizer
from tcga_bench.training.vicreg import vicreg_loss


def test_tokenizer_shape():
    tok = FeatureTokenizer(n_features=20, dim=8)
    x = torch.randn(4, 20)
    out = tok(x)
    assert out.shape == (4, 20, 8)


def test_tokenizer_permutation_equivariance():
    """tok(x)[:, perm, :] == tok'(x[:, perm]) where tok' carries the same
    feature-identity embeddings, reindexed by perm. The embedding must
    travel with the permutation (see tokenizer.py docstring) -- a plain
    tok(x[:, perm]) with the embedding table left untouched is NOT
    equivariant, since position i would then be paired with the wrong
    identity embedding."""
    torch.manual_seed(0)
    n_features, dim = 16, 8
    tok = FeatureTokenizer(n_features, dim)
    x = torch.randn(2, n_features)
    perm = torch.randperm(n_features)

    tok_perm = FeatureTokenizer(n_features, dim)
    tok_perm.load_state_dict(tok.state_dict())
    with torch.no_grad():
        tok_perm.feature_embed.copy_(tok.feature_embed[perm])

    out = tok(x)
    out_permuted = tok_perm(x[:, perm])

    assert torch.allclose(out[:, perm, :], out_permuted, atol=1e-5)


def test_backbone_forward_shapes():
    backbone = TabularSetBackbone(n_features=32, dim=16, n_heads=2, n_inducing=8, n_isab_layers=2)
    x = torch.randn(5, 32)
    rep = backbone(x)
    assert rep.shape == (5, 16)


def test_plain_mlp_backbone_shapes():
    backbone = PlainMLPBackbone(n_features=32, dim=16)
    x = torch.randn(5, 32)
    assert backbone(x).shape == (5, 16)


def test_ssl_projector_shape():
    proj = SSLProjector(in_dim=16, hidden_dim=32, out_dim=24)
    x = torch.randn(6, 16)
    assert proj(x).shape == (6, 24)


def test_vicreg_loss_is_finite_and_decreases_on_identical_views():
    z = torch.randn(32, 24)
    loss_same, parts = vicreg_loss(z, z.clone())
    assert torch.isfinite(loss_same)
    assert parts["invariance"] == 0.0

    z2 = torch.randn(32, 24) * 5
    loss_diff, _ = vicreg_loss(z, z2)
    assert torch.isfinite(loss_diff)


def test_bayesian_linear_kl_nonnegative_and_sampling_varies():
    head = BayesianLinear(in_dim=16, out_dim=4)
    kl = head.kl_divergence()
    assert kl.item() >= 0

    x = torch.randn(10, 16)
    out1 = head(x, sample=True)
    out2 = head(x, sample=True)
    assert not torch.allclose(out1, out2)  # stochastic weights -> different outputs

    out_det = head(x, sample=False)
    assert out_det.shape == (10, 4)


def test_bayesian_predict_distribution_shape():
    head = BayesianLinear(in_dim=8, out_dim=3)
    x = torch.randn(5, 8)
    probs = head.predict_distribution(x, n_samples=7)
    assert probs.shape == (7, 5, 3)
    assert torch.allclose(probs.sum(dim=-1), torch.ones(7, 5), atol=1e-4)
