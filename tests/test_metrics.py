import numpy as np

from prismwf.metrics import fixed_k_metrics, unknown_k_metrics


def test_fixed_k_and_unknown_k_metrics() -> None:
    labels = np.asarray([[1, 1, 0], [0, 1, 1], [1, 0, 1]], dtype=np.int64)
    scores = np.asarray(
        [[0.9, 0.8, 0.1], [0.1, 0.8, 0.9], [0.9, 0.1, 0.8]],
        dtype=np.float64,
    )

    fixed = fixed_k_metrics(labels, scores, 2)
    assert fixed["P@2"] == 1.0
    assert fixed["MAP@2"] == 1.0

    unknown = unknown_k_metrics(labels, scores, 0.5)
    assert unknown["micro_f1"] == 1.0
    assert unknown["exact_set_accuracy"] == 1.0
    assert unknown["label_count_mae"] == 0.0
