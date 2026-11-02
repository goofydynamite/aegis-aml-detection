"""Check that the live feature engine matches the batch training features on real data.

Replays every transaction through FeatureEngine and compares the last --rows of them with
the features file, feature by feature.

Usage: python scripts/check_online_parity.py [--config config.yaml] [--rows 300000]
"""
from __future__ import annotations

import argparse
import json
import math
import time

import polars as pl

from aegis.config import Config
from aegis.features import FEATURES
from aegis.online import REPLAY_COLUMNS, new_engine, replay_rows

TOLERANCE = 1e-4  # relative; the features file stores 32-bit floats

parser = argparse.ArgumentParser()
parser.add_argument("--config", default="config.yaml")
parser.add_argument("--rows", type=int, default=300_000)
args = parser.parse_args()

cfg = Config.load(args.config)
tx = pl.read_parquet(cfg.transactions_parquet, columns=REPLAY_COLUMNS)
engine = new_engine(tx)
tx = tx.drop("from_bank", "from_account", "to_bank", "to_account")
start_compare = len(tx) - args.rows
expected = pl.read_parquet(cfg.features_parquet, columns=FEATURES).slice(start_compare).to_dicts()

started = time.time()
for batch in tx.slice(0, start_compare).iter_slices(500_000):
    for row in replay_rows(batch):
        engine.process_ids(*row)
    print(f"  {engine.transactions:,} replayed, {time.time() - started:.0f}s", flush=True)
print(f"Replayed {engine.transactions:,} payments in {time.time() - started:.0f}s", flush=True)

mismatches = {f: 0 for f in FEATURES}
examples: dict[str, tuple] = {}
for i, row in enumerate(replay_rows(tx.slice(start_compare))):
    values = engine.process_ids(*row)
    want = expected[i]
    for f in FEATURES:
        a, b = values[f], want[f]
        a_missing = a is None or (isinstance(a, float) and math.isnan(a))
        b_missing = b is None or (isinstance(b, float) and math.isnan(b))
        same = a_missing and b_missing
        if not a_missing and not b_missing:
            same = abs(a - b) <= TOLERANCE * max(abs(a), abs(b), 1.0)
        if not same:
            mismatches[f] += 1
            examples.setdefault(f, (a, b))

report = {f: round(n / args.rows, 6) for f, n in mismatches.items()}
(cfg.reports_dir / "online_parity.json").write_text(json.dumps(
    {"rows_compared": args.rows, "tolerance": TOLERANCE, "mismatch_rate": report}, indent=2), encoding="utf-8")
print(f"{'feature':<28} mismatch rate  example (live, batch)")
for f in sorted(FEATURES, key=lambda x: -mismatches[x]):
    print(f"{f:<28} {report[f]:>10.4%}     {examples.get(f, '')}")
