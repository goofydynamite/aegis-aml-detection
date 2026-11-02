"""Group alerts into investigation cases and roll risk up to accounts.

Alerts that share an account belong to one case, so an investigator sees a network
instead of a list of separate payments.
"""
from __future__ import annotations

import polars as pl


def assign_cases(alerts: pl.DataFrame) -> pl.DataFrame:
    """Add case_id: connected components of the graph formed by alerted payments."""
    parent: dict[str, str] = {}

    def find(x: str) -> str:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    senders = (alerts["from_bank"] + "/" + alerts["from_account"]).to_list()
    receivers = (alerts["to_bank"] + "/" + alerts["to_account"]).to_list()
    for s, r in zip(senders, receivers):
        parent[find(s)] = find(r)

    roots = [find(s) for s in senders]
    df = alerts.with_columns(
        sender_key=pl.Series(senders), receiver_key=pl.Series(receivers), root=pl.Series(roots)
    )
    # numbered by size, so case 1 is the largest
    order = (
        df.group_by("root").agg(n=pl.len(), top=pl.col("risk_score").max())
        .sort(["n", "top"], descending=True)
        .with_row_index("case_id", offset=1)
        .select("root", pl.col("case_id").cast(pl.Int64))
    )
    return df.join(order, on="root").drop("root")


def summarise_cases(alerts: pl.DataFrame) -> pl.DataFrame:
    return (
        alerts.group_by("case_id")
        .agg(
            alerts=pl.len(),
            accounts=pl.concat_list("sender_key", "receiver_key").explode().n_unique(),
            total_usd=pl.col("amount_usd").sum().round(2),
            max_risk=pl.col("risk_score").max(),
            first_seen=pl.col("timestamp").min(),
            last_seen=pl.col("timestamp").max(),
            confirmed_laundering=pl.col("is_laundering").sum(),
        )
        .sort("case_id")
    )


def account_risk(alerts: pl.DataFrame) -> pl.DataFrame:
    """Per-account roll-up across the alert queue."""
    sides = pl.concat([
        alerts.select(account="sender_key", role=pl.lit("sender"), risk_score="risk_score",
                      amount_usd="amount_usd", case_id="case_id"),
        alerts.select(account="receiver_key", role=pl.lit("receiver"), risk_score="risk_score",
                      amount_usd="amount_usd", case_id="case_id"),
    ])
    return (
        sides.group_by("account")
        .agg(
            alerts=pl.len(),
            as_sender=(pl.col("role") == "sender").sum(),
            as_receiver=(pl.col("role") == "receiver").sum(),
            max_risk=pl.col("risk_score").max(),
            total_usd=pl.col("amount_usd").sum().round(2),
            case_id=pl.col("case_id").first(),
        )
        .sort(["max_risk", "alerts"], descending=True)
    )
