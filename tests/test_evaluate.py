import numpy as np
import pytest

from aegis.evaluate import pattern_recall, precision_recall_at_k


def test_precision_recall_at_k():
    y = np.array([1, 0, 1, 0, 0, 1])
    scores = np.array([0.9, 0.8, 0.7, 0.1, 0.2, 0.05])
    p, r = precision_recall_at_k(y, scores, 3)
    assert p == pytest.approx(2 / 3)
    assert r == pytest.approx(2 / 3)


def test_pattern_recall_counts_hits_in_top_k():
    patterns = np.array(["CYCLE", None, "CYCLE", "FAN-IN"], dtype=object)
    scores = np.array([0.9, 0.1, 0.2, 0.8])
    out = pattern_recall(patterns, scores, k=2)
    assert out["CYCLE"] == {"count": 2, "caught": 1, "recall": 0.5}
    assert out["FAN-IN"]["recall"] == 1.0
