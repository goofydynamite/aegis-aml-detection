"""Behavioural and network features computed from each account's history.

Every feature for a transaction uses only that transaction and earlier ones, so the
same code can score a live stream. Input must be sorted by (timestamp, txn_id).
Within a minute, rows count as ordered by txn_id. online.py computes the same values one
payment at a time, and scripts/check_online_parity.py compares the two on real data.
"""
from __future__ import annotations

import polars as pl

from .cycles import CYCLE_FEATURES, add_cycle_features
from .graph import GRAPH_FEATURES, add_graph_features

WINDOW = "24h"

FEATURES = [
    # the transaction itself
    "log_amount_usd", "is_cross_currency", "is_self_transfer", "is_round_amount",
    "hour", "weekday", "payment_format_code",
    # sender's outgoing behaviour
    "s_out_prior_count", "s_out_count_24h", "s_out_sum_24h", "s_distinct_receivers",
    "s_secs_since_prev_out", "amount_vs_sender_mean", "s_is_new_receiver",
    # receiver's incoming behaviour
    "r_in_prior_count", "r_in_count_24h", "r_in_sum_24h", "r_distinct_senders",
    "r_secs_since_prev_in", "amount_vs_receiver_mean",
    # cross-role: money flowing through the sender, receiver acting as a hub
    "s_secs_since_last_in", "s_last_in_amount_ratio", "s_in_distinct_senders",
    "s_in_count_24h_at_last_in", "r_out_distinct_receivers", "r_out_prior_count",
    *CYCLE_FEATURES,
]
CATEGORICAL_FEATURES = ["payment_format_code"]

# GRAPH_FEATURES are built but left out of FEATURES: they added nothing on validation data
# (scripts/compare_feature_sets.py).

# The legacy prototype's feature set, kept as a baseline for comparison.
LEGACY_FEATURES = ["log_amount_usd", "is_odd_hour", "is_round_amount"]


def _prior_count(by: str | list[str]) -> pl.Expr:
    return pl.int_range(pl.len(), dtype=pl.UInt32).over(by)


def _window(col: pl.Expr, by: str) -> pl.Expr:
    # Rolling windows see the whole of the current minute; take back the rows after this one.
    later_this_minute = (col.sum() - col.cum_sum()).over(by, "timestamp")
    return col.rolling_sum_by("timestamp", window_size=WINDOW).over(by) - later_this_minute


def build_features(tx: pl.DataFrame) -> pl.DataFrame:
    amount = pl.col("amount_usd")
    one = amount.is_not_null().cast(pl.UInt32)

    df = tx.with_columns(
        log_amount_usd=amount.log1p(),
        is_cross_currency=(pl.col("payment_currency") != pl.col("receiving_currency")).cast(pl.Int8),
        is_self_transfer=(pl.col("sender") == pl.col("receiver")).cast(pl.Int8),
        is_round_amount=((pl.col("amount_paid") % 100) == 0).cast(pl.Int8),
        hour=pl.col("timestamp").dt.hour().cast(pl.Int8),
        weekday=pl.col("timestamp").dt.weekday().cast(pl.Int8),
        payment_format_code=pl.col("payment_format").to_physical().cast(pl.Int8),
        # sender as payer
        s_out_prior_count=_prior_count("sender"),
        s_out_prior_sum=(amount.cum_sum() - amount).over("sender"),
        s_out_count_24h=_window(one, "sender"),
        s_out_sum_24h=_window(amount, "sender"),
        s_secs_since_prev_out=pl.col("timestamp").diff().dt.total_seconds().over("sender"),
        s_is_new_receiver=(_prior_count(["sender", "receiver"]) == 0).cast(pl.Int8),
        # receiver as payee
        r_in_prior_count=_prior_count("receiver"),
        r_in_prior_sum=(amount.cum_sum() - amount).over("receiver"),
        r_in_count_24h=_window(one, "receiver"),
        r_in_sum_24h=_window(amount, "receiver"),
        r_secs_since_prev_in=pl.col("timestamp").diff().dt.total_seconds().over("receiver"),
        r_is_new_sender=(_prior_count(["receiver", "sender"]) == 0).cast(pl.Int8),
    ).with_columns(
        s_distinct_receivers=pl.col("s_is_new_receiver").cast(pl.UInt32).cum_sum().over("sender"),
        r_distinct_senders=pl.col("r_is_new_sender").cast(pl.UInt32).cum_sum().over("receiver"),
        amount_vs_sender_mean=amount / (pl.col("s_out_prior_sum") / pl.col("s_out_prior_count")),
        amount_vs_receiver_mean=amount / (pl.col("r_in_prior_sum") / pl.col("r_in_prior_count")),
        is_odd_hour=((pl.col("hour") < 6) | (pl.col("hour") > 22)).cast(pl.Int8),
    )

    # What had the sender received before paying out? (pass-through, stacks, gather nodes)
    inflows = df.select(
        pl.col("receiver").alias("sender"),
        pl.col("timestamp").alias("last_in_ts"),
        pl.col("amount_usd").alias("last_in_amount"),
        pl.col("r_distinct_senders").alias("s_in_distinct_senders"),
        pl.col("r_in_count_24h").alias("s_in_count_24h_at_last_in"),
    )
    # What had the receiver paid out before? (fan-out hubs, scatter nodes)
    outflows = df.select(
        pl.col("sender").alias("receiver"),
        pl.col("timestamp").alias("last_out_ts"),
        pl.col("s_distinct_receivers").alias("r_out_distinct_receivers"),
        (pl.col("s_out_prior_count") + 1).alias("r_out_prior_count"),
    )
    # allow_exact_matches=False keeps events from the same minute out of the lookup.
    df = (
        df.join_asof(inflows, left_on="timestamp", right_on="last_in_ts", by="sender",
                     strategy="backward", allow_exact_matches=False, check_sortedness=False)
        .join_asof(outflows, left_on="timestamp", right_on="last_out_ts", by="receiver",
                   strategy="backward", allow_exact_matches=False, check_sortedness=False)
        .with_columns(
            s_secs_since_last_in=(pl.col("timestamp") - pl.col("last_in_ts")).dt.total_seconds(),
            s_last_in_amount_ratio=amount / pl.col("last_in_amount"),
        )
    )

    df = add_cycle_features(add_graph_features(df))

    keep = ["txn_id", "timestamp", "is_laundering", "pattern"] + FEATURES + GRAPH_FEATURES + ["is_odd_hour"]
    numeric = [c for c in FEATURES + GRAPH_FEATURES
               if c not in CATEGORICAL_FEATURES and c not in ("hour", "weekday")]
    return df.select(keep).with_columns(
        # ratios against an empty or zero history give NaN/inf; treat them as missing
        pl.when(pl.col(c).cast(pl.Float32).is_infinite() | pl.col(c).cast(pl.Float32).is_nan()).then(None)
        .otherwise(pl.col(c).cast(pl.Float32)).alias(c)
        for c in numeric
    )
