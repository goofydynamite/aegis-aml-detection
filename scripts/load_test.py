"""Measure the speed of a running API with real test-period payments.

This measures speed only. Timestamps are moved past the latest payment the service has
seen, so to the service these are repeats, and their scores say nothing about detection.
Each run adds the payments to the service's history.

Usage: python scripts/load_test.py [--url http://127.0.0.1:8000] [--requests 500]
"""
from __future__ import annotations

import argparse
import statistics
import time
from datetime import datetime, timedelta

import httpx
import polars as pl

parser = argparse.ArgumentParser()
parser.add_argument("--url", default="http://127.0.0.1:8000")
parser.add_argument("--requests", type=int, default=500)
parser.add_argument("--reports", default="reports")
args = parser.parse_args()

scored = pl.read_parquet(f"{args.reports}/scored_test.parquet")
rows = scored.sample(args.requests, seed=1).to_dicts()

client = httpx.Client(base_url=args.url, timeout=30)
latest = client.get("/health").json()["latest_transaction"]
start = datetime.fromisoformat(latest) + timedelta(minutes=1)

times = []
for i, row in enumerate(rows):
    began = time.perf_counter()
    response = client.post("/score", json={
        "timestamp": (start + timedelta(seconds=i)).isoformat(),
        "from_bank": row["from_bank"], "from_account": row["from_account"],
        "to_bank": row["to_bank"], "to_account": row["to_account"],
        "amount_paid": row["amount_paid"], "payment_currency": row["payment_currency"],
        "receiving_currency": row["payment_currency"], "payment_format": str(row["payment_format"]),
    })
    times.append((time.perf_counter() - began) * 1000)
    response.raise_for_status()

times = times[20:]  # the first calls warm up the service
q = statistics.quantiles(times, n=100)
print(f"latency ms: median {statistics.median(times):.1f}, p95 {q[94]:.1f}, p99 {q[98]:.1f}, "
      f"max {max(times):.1f}; about {1000 / statistics.mean(times):.0f} requests/s one at a time")
