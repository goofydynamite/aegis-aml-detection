"""Load the IBM AML (AMLworld) transaction files and convert them to Parquet."""
from __future__ import annotations

import re
from pathlib import Path

import polars as pl

# The raw header repeats "Account" for sender and receiver, so names are set here.
RAW_SCHEMA = {
    "timestamp": pl.String,
    "from_bank": pl.String,
    "from_account": pl.String,
    "to_bank": pl.String,
    "to_account": pl.String,
    "amount_received": pl.Float64,
    "receiving_currency": pl.String,
    "amount_paid": pl.Float64,
    "payment_currency": pl.String,
    "payment_format": pl.String,
    "is_laundering": pl.Int8,
}
PAYMENT_FORMATS = ["ACH", "Bitcoin", "Cash", "Cheque", "Credit Card", "Reinvestment", "Wire"]
PAYMENT_FORMAT_DTYPE = pl.Enum(PAYMENT_FORMATS)
TS_FORMAT = "%Y/%m/%d %H:%M"
USD = "US Dollar"

# Columns that identify a transaction when matching the patterns file.
_MATCH_KEYS = ["timestamp", "from_bank", "from_account", "to_bank", "to_account",
               "amount_paid", "payment_currency", "payment_format"]
_BEGIN = re.compile(r"^BEGIN LAUNDERING ATTEMPT - ([A-Z\-]+)")


def scan_raw_transactions(path: str | Path) -> pl.LazyFrame:
    return pl.scan_csv(path, has_header=False, skip_rows=1, schema=RAW_SCHEMA)


def read_patterns(path: str | Path) -> pl.DataFrame:
    """Parse the *_Patterns.txt file into one row per transaction with its typology."""
    rows: list[list[str]] = []
    current: str | None = None
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if m := _BEGIN.match(line):
            current = m.group(1)
        elif line.startswith("END") or not line.strip():
            current = None
        elif current:
            rows.append(line.split(",") + [current])

    names = list(RAW_SCHEMA) + ["pattern"]
    df = pl.DataFrame(rows, schema=names, orient="row")
    return (
        df.with_columns(pl.col("amount_paid").cast(pl.Float64))
        .select(_MATCH_KEYS + ["pattern"])
        .unique(subset=_MATCH_KEYS, keep="first", maintain_order=True)
    )


def usd_rates(lf: pl.LazyFrame) -> dict[str, float]:
    """Units of each currency per US dollar, estimated from cross-currency payments."""
    rates = (
        lf.filter(
            (pl.col("payment_currency") != pl.col("receiving_currency"))
            & ((pl.col("payment_currency") == USD) | (pl.col("receiving_currency") == USD))
            & (pl.col("amount_paid") > 0)
            & (pl.col("amount_received") > 0)
        )
        .with_columns(
            currency=pl.when(pl.col("payment_currency") == USD)
            .then(pl.col("receiving_currency"))
            .otherwise(pl.col("payment_currency")),
            rate=pl.when(pl.col("payment_currency") == USD)
            .then(pl.col("amount_received") / pl.col("amount_paid"))
            .otherwise(pl.col("amount_paid") / pl.col("amount_received")),
        )
        .group_by("currency")
        .agg(pl.col("rate").median())
        .collect()
    )
    out = dict(zip(rates["currency"].to_list(), rates["rate"].to_list()))
    out[USD] = 1.0
    return out


def trim_sparse_tail(df: pl.DataFrame, min_day_share: float) -> pl.DataFrame:
    """Drop trailing days whose volume collapses below min_day_share of the median day.

    The AMLworld files end with a few near-empty days that are mostly laundering
    (attempts that run past the simulation end). Keeping them lets a model learn
    "quiet period => laundering", which inflates test scores.
    """
    days = (
        df.group_by(day=pl.col("timestamp").dt.date()).agg(rows=pl.len()).sort("day")
    )
    threshold = days["rows"].median() * min_day_share
    full = days.filter(pl.col("rows") >= threshold)
    return df.filter(pl.col("timestamp").dt.date() <= full["day"].max())


def _account_ids(df: pl.DataFrame) -> pl.DataFrame:
    keys = pl.concat([
        df.select(key=pl.col("from_bank") + "/" + pl.col("from_account")),
        df.select(key=pl.col("to_bank") + "/" + pl.col("to_account")),
    ]).unique().sort("key")
    return keys.with_row_index("account_id")


def ingest(raw_csv: str | Path, patterns_txt: str | Path | None, out_path: str | Path,
           min_day_share: float = 0.01) -> pl.DataFrame:
    """Clean the raw CSV, attach typology labels and save a time-sorted Parquet file."""
    lf = scan_raw_transactions(raw_csv)
    rates = usd_rates(lf)

    df = lf.unique(maintain_order=True).collect()
    if patterns_txt and Path(patterns_txt).exists():
        df = df.join(read_patterns(patterns_txt), on=_MATCH_KEYS, how="left")
    else:
        df = df.with_columns(pattern=pl.lit(None, dtype=pl.String))

    accounts = _account_ids(df)
    df = (
        df.with_columns(
            sender_key=pl.col("from_bank") + "/" + pl.col("from_account"),
            receiver_key=pl.col("to_bank") + "/" + pl.col("to_account"),
        )
        .join(accounts.rename({"key": "sender_key", "account_id": "sender"}), on="sender_key", how="left")
        .join(accounts.rename({"key": "receiver_key", "account_id": "receiver"}), on="receiver_key", how="left")
        .with_columns(
            timestamp=pl.col("timestamp").str.to_datetime(TS_FORMAT),
            payment_format=pl.col("payment_format").cast(PAYMENT_FORMAT_DTYPE),
            amount_usd=pl.col("amount_paid")
            / pl.col("payment_currency").replace_strict(rates, default=None, return_dtype=pl.Float64),
        )
        .sort("timestamp", maintain_order=True)
        .pipe(trim_sparse_tail, min_day_share)
        .with_row_index("txn_id")
        .select(
            "txn_id", "timestamp", "sender", "receiver",
            "from_bank", "from_account", "to_bank", "to_account",
            "amount_paid", "payment_currency", "amount_received", "receiving_currency",
            "amount_usd", "payment_format", "is_laundering", "pattern",
        )
    )
    df.write_parquet(out_path, compression="zstd")
    return df
