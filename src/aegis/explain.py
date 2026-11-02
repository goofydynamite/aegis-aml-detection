"""Plain-language reasons for each alert, from LightGBM's per-feature contributions."""
from __future__ import annotations

import lightgbm as lgb
import numpy as np

REASON_LABELS = {
    "log_amount_usd": "payment size",
    "is_cross_currency": "currency conversion",
    "is_self_transfer": "payment to own account",
    "is_round_amount": "round amount",
    "hour": "time of day",
    "weekday": "day of week",
    "payment_format_code": "payment method",
    "s_out_prior_count": "sender's payment history length",
    "s_out_count_24h": "sender's payments in last 24h",
    "s_out_sum_24h": "sender's volume in last 24h",
    "s_distinct_receivers": "number of accounts the sender has paid",
    "s_secs_since_prev_out": "time since sender's last payment",
    "amount_vs_sender_mean": "amount unusual for this sender",
    "s_is_new_receiver": "first payment to this receiver",
    "r_in_prior_count": "receiver's payment history length",
    "r_in_count_24h": "receiver's incoming payments in last 24h",
    "r_in_sum_24h": "receiver's incoming volume in last 24h",
    "r_distinct_senders": "number of accounts paying the receiver",
    "r_secs_since_prev_in": "time since receiver last got paid",
    "amount_vs_receiver_mean": "amount unusual for this receiver",
    "s_secs_since_last_in": "time since sender received funds",
    "s_last_in_amount_ratio": "payment matches sender's last inflow",
    "s_in_distinct_senders": "number of accounts paying the sender",
    "s_in_count_24h_at_last_in": "recent inflows into sender",
    "r_out_distinct_receivers": "number of accounts the receiver pays",
    "r_out_prior_count": "receiver's outgoing payment history",
    "secs_since_reverse_payment": "money recently came back the other way",
    "cycle_length": "payment closes a cycle of accounts",
    "cycle_secs": "speed of the cycle this payment closes",
    "chain_depth": "money reached the sender through a long chain",
    "chain_sources": "many upstream sources feed the sender",
    "s_max_parallel_paths": "sender fans out through parallel routes",
    "r_max_parallel_paths": "receiver gathers from parallel routes",
}


def top_reasons(booster: lgb.Booster, X: np.ndarray, features: list[str], n: int = 3) -> list[str]:
    """For each row, the n features that pushed its risk score up the most."""
    contrib = booster.predict(X, pred_contrib=True, num_iteration=booster.best_iteration)[:, :-1]
    order = np.argsort(-contrib, axis=1)[:, :n]
    return [
        "; ".join(REASON_LABELS[features[j]] for j in row if contrib[i, j] > 0)
        for i, row in enumerate(order)
    ]
