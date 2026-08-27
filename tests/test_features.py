import numpy as np

from prismwf.data import FEATURE_GROUPS
from prismwf.features import _encode_one, slot_features
from tools.split_ares_dataset import split_indices


def test_slot_features_counts_transitions_and_intervals() -> None:
    packets = np.asarray([1.0, 2.0, -4.0, -5.0, 8.0], dtype=np.float64)
    actual = slot_features(packets)
    expected = np.asarray([3.0, 2.0, 1.0, 1.0, 2.0, 3.0], dtype=np.float32)
    np.testing.assert_allclose(actual, expected)


def test_feature_groups_partition_six_channels() -> None:
    selected = (
        set(FEATURE_GROUPS["packet-count"])
        | set(FEATURE_GROUPS["transition-count"])
        | set(FEATURE_GROUPS["transition-interval"])
    )
    assert selected == set(FEATURE_GROUPS["all"])


def test_strict_window_discards_tail_events() -> None:
    trace_ms = np.asarray([1.0, 5.0, 25.0], dtype=np.float64)
    _, compatible = _encode_one(0, trace_ms, 10, 2, True)
    _, strict = _encode_one(0, trace_ms, 10, 2, False)
    assert compatible[0].sum() == 3
    assert strict[0].sum() == 2


def test_wflib_split_is_deterministic_and_disjoint() -> None:
    train, valid, test = split_indices(100, seed=2024)
    assert (len(train), len(valid), len(test)) == (81, 9, 10)
    merged = np.concatenate([train, valid, test])
    assert len(np.unique(merged)) == 100
    repeated = split_indices(100, seed=2024)
    for actual, expected in zip(repeated, (train, valid, test)):
        np.testing.assert_array_equal(actual, expected)
