import numpy as np

from tcga_bench.eval.calibration import expected_calibration_error, predictive_entropy


def test_ece_zero_when_perfectly_calibrated():
    # 100 predictions, all with confidence 0.8, exactly 80% correct
    n = 100
    probs = np.tile([0.8, 0.2], (n, 1))
    labels = np.array([0] * 80 + [1] * 20)
    ece, bins = expected_calibration_error(probs, labels, n_bins=10)
    assert ece < 1e-6
    assert len(bins) == 10


def test_ece_positive_when_overconfident():
    n = 100
    probs = np.tile([0.99, 0.01], (n, 1))
    labels = np.array([0] * 50 + [1] * 50)  # only 50% accurate despite 99% confidence
    ece, _ = expected_calibration_error(probs, labels, n_bins=10)
    assert ece > 0.4


def test_predictive_entropy_uniform_is_max():
    probs_uniform = np.full((1, 4), 0.25)
    probs_confident = np.array([[0.97, 0.01, 0.01, 0.01]])
    ent_uniform = predictive_entropy(probs_uniform)[0]
    ent_confident = predictive_entropy(probs_confident)[0]
    assert ent_uniform > ent_confident
