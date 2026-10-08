"""Calibration diagnostics for a probabilistic classifier: expected
calibration error (ECE) and reliability-diagram bins. Used to check
whether the Bayesian head's predictive probabilities are honest, not
just whether argmax accuracy is high."""
import numpy as np


def expected_calibration_error(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15):
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    accuracies = (predictions == labels).astype(np.float32)

    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    bins = []
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        in_bin = (confidences > lo) & (confidences <= hi) if i > 0 else (confidences >= lo) & (confidences <= hi)
        if in_bin.sum() == 0:
            bins.append({"bin_lo": float(lo), "bin_hi": float(hi), "count": 0, "avg_confidence": None, "avg_accuracy": None})
            continue
        bin_conf = confidences[in_bin].mean()
        bin_acc = accuracies[in_bin].mean()
        weight = in_bin.sum() / len(confidences)
        ece += weight * abs(bin_acc - bin_conf)
        bins.append({
            "bin_lo": float(lo), "bin_hi": float(hi), "count": int(in_bin.sum()),
            "avg_confidence": float(bin_conf), "avg_accuracy": float(bin_acc),
        })
    return float(ece), bins


def predictive_entropy(probs: np.ndarray) -> np.ndarray:
    eps = 1e-12
    return -(probs * np.log(probs + eps)).sum(axis=1)
