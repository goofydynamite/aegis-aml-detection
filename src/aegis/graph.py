"""Network features built from short money paths between accounts.

Paths are a -> mid -> c, where the second hop follows the first within PATH_WINDOW.
Each transaction looks up paths that finished strictly before it, so nothing depends on
later payments. Accounts above hub_cap counterparties are left out of the paths: they
are payment processors, where a shared neighbour says nothing. That cap is the one
input measured over the whole file.
"""
from __future__ import annotations

from datetime import timedelta

import polars as pl

PATH_WINDOW = timedelta(hours=72)
GRAPH_FEATURES = [
    "secs_since_reverse_payment",
    "s_max_parallel_paths", "r_max_parallel_paths",
]


def hub_accounts(tx: pl.DataFrame, hub_cap: int) -> pl.DataFrame:
    """Accounts with more than hub_cap distinct counterparties on either side."""
    distinct = tx.select("sender", "receiver").filter(pl.col("sender") != pl.col("receiver")).unique()
    return pl.concat([
        distinct.group_by("sender").len().filter(pl.col("len") > hub_cap).select(account="sender"),
        distinct.group_by("receiver").len().filter(pl.col("len") > hub_cap).select(account="receiver"),
    ]).unique()


def _daily_edges(tx: pl.DataFrame, hub_cap: int) -> pl.DataFrame:
    """Non-self edges between non-hub accounts, reduced to their first use each day."""
    edges = tx.select("sender", "receiver", "timestamp").filter(pl.col("sender") != pl.col("receiver"))
    hubs = hub_accounts(tx, hub_cap)
    return (
        edges.join(hubs, left_on="sender", right_on="account", how="anti")
        .join(hubs, left_on="receiver", right_on="account", how="anti")
        .group_by("sender", "receiver", pl.col("timestamp").dt.date().alias("day"))
        .agg(ts=pl.col("timestamp").min())
        .drop("day")
    )


def two_hop_paths(edges: pl.DataFrame) -> pl.DataFrame:
    """First completion of each distinct path, with the parallel-route count so far."""
    return (
        edges.rename({"sender": "a", "receiver": "mid", "ts": "t1"})
        .join(edges.rename({"sender": "mid", "receiver": "c", "ts": "t2"}), on="mid")
        .filter(
            (pl.col("a") != pl.col("c"))
            & (pl.col("t2") > pl.col("t1"))
            & (pl.col("t2") - pl.col("t1") <= PATH_WINDOW)
        )
        .group_by("a", "mid", "c")
        .agg(t2=pl.col("t2").min())
        .sort("t2")
        .with_columns(parallel=pl.int_range(1, pl.len() + 1, dtype=pl.UInt32).over("a", "c"))
    )


def add_graph_features(df: pl.DataFrame, hub_cap: int = 1000) -> pl.DataFrame:
    """Expects df sorted by timestamp, with sender, receiver and timestamp columns."""
    edges = _daily_edges(df, hub_cap)
    paths = two_hop_paths(edges)

    reverse = edges.sort("ts").select(
        pl.col("receiver").alias("rev_to"), pl.col("sender").alias("rev_from"),
        pl.col("ts").alias("reverse_ts"),
    )
    by_origin = paths.select(
        pl.col("a").alias("sender"), pl.col("t2").alias("origin_ts"),
        pl.col("parallel").cum_max().over("a").alias("s_max_parallel_paths"),
    )
    by_target = paths.select(
        pl.col("c").alias("receiver"), pl.col("t2").alias("target_ts"),
        pl.col("parallel").cum_max().over("c").alias("r_max_parallel_paths"),
    )

    asof = dict(strategy="backward", allow_exact_matches=False, check_sortedness=False)
    out = (
        df.join_asof(reverse, left_on="timestamp", right_on="reverse_ts",
                     by_left=["sender", "receiver"], by_right=["rev_to", "rev_from"], **asof)
        .join_asof(by_origin, left_on="timestamp", right_on="origin_ts", by="sender", **asof)
        .join_asof(by_target, left_on="timestamp", right_on="target_ts", by="receiver", **asof)
    )
    return out.with_columns(
        secs_since_reverse_payment=(pl.col("timestamp") - pl.col("reverse_ts")).dt.total_seconds(),
    ).drop("reverse_ts", "origin_ts", "target_ts")
