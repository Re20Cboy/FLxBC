import numpy as np

from flxbc.training import _classification_metrics


def test_classification_metrics_include_cross_entropy_loss():
    probabilities = np.array(
        [
            [0.9, 0.1],
            [0.2, 0.8],
            [0.6, 0.4],
        ],
        dtype=np.float64,
    )
    predictions = np.array([0, 1, 0], dtype=np.int64)
    labels = np.array([0, 1, 1], dtype=np.int64)

    metrics = _classification_metrics(predictions, labels, 2, probabilities=probabilities)

    expected_loss = -float(np.mean(np.log([0.9, 0.8, 0.4])))
    assert metrics["loss"] == expected_loss
    assert metrics["accuracy"] == 2 / 3
    assert "macro_f1" in metrics
