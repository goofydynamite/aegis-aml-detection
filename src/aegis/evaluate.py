"""Ranking metrics for alert queues.

Laundering is ~0.05% of transactions, so accuracy is meaningless. An investigation team
reviews a fixed number of alerts, so the questions are: of the top-k alerts, how many are
laundering (precision@k), and what share of all laundering do they contain (recall@k)?
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def precision_recall_at_k(y_true: np.ndarray, scores: np.ndarray, k: int) -> tuple[float, float]:
    k = min(k, len(scores))
    top = np.argpartition(-scores, k - 1)[:k]
    hits = int(y_true[top].sum())
    positives = int(y_true.sum())
    return hits / k, (hits / positives if positives else float("nan"))


def evaluate_scores(y_true: np.ndarray, scores: np.ndarray, k_values: list[int]) -> dict:
    y_true = np.asarray(y_true).astype(int)
    out = {
        "n": int(len(y_true)),
        "positives": int(y_true.sum()),
        "pr_auc": float(average_precision_score(y_true, scores)),
        "roc_auc": float(roc_auc_score(y_true, scores)),
        "base_rate": float(y_true.mean()),
    }
    for k in k_values:
        p, r = precision_recall_at_k(y_true, scores, k)
        out[f"precision@{k}"] = p
        out[f"recall@{k}"] = r
    return out


def pattern_recall(patterns: np.ndarray, scores: np.ndarray, k: int) -> dict[str, dict]:
    """Recall within the top-k alerts for each laundering typology."""
    k = min(k, len(scores))
    in_top = np.zeros(len(scores), dtype=bool)
    in_top[np.argpartition(-scores, k - 1)[:k]] = True
    out = {}
    for name in sorted({p for p in patterns if p is not None}):
        mask = patterns == name
        out[name] = {"count": int(mask.sum()), "caught": int(in_top[mask].sum()),
                     "recall": float(in_top[mask].mean())}
    return out
