"""How much a "closed cycle always alerts" rule adds on top of the model's alerts.

The model alerts on the same share of payments as its validation threshold (top --alerts of
validation). For each minimum cycle length the rule adds alerts for payments that close a
cycle at least that long, and this reports how many extra alerts it makes and how many are
laundering.

Usage: python scripts/evaluate_cycle_rule.py --split valid [--config config.yaml]
"""
from __future__ import annotations

import argparse
import json

import lightgbm as lgb
import numpy as np
import polars as pl

from aegis import features, model
from aegis.config import Config

parser = argparse.ArgumentParser()
parser.add_argument("--config", default="config.yaml")
parser.add_argument("--model", help="defaults to the config's own model")
parser.add_argument("--split", choices=["valid", "test"], default="valid")
parser.add_argument("--lengths", type=int, nargs="+", default=[2, 3, 4, 5, 6])
args = parser.parse_args()

cfg = Config.load(args.config)
model_path = args.model or str(cfg.models_dir / f"{cfg.dataset}_lightgbm.txt")
feats = pl.read_parquet(cfg.features_parquet, columns=["timestamp", "is_laundering", *features.FEATURES])
_, valid, test = model.time_split(feats, **cfg["split"])
del feats
booster = lgb.Booster(model_file=model_path)

k = cfg["evaluation"]["top_alerts_export"]
threshold = float(np.sort(booster.predict(model.to_matrix(valid, features.FEATURES)))[-k])
part = valid if args.split == "valid" else test
scores = booster.predict(model.to_matrix(part, features.FEATURES))
y = part["is_laundering"].to_numpy()
cycle = part["cycle_length"].fill_null(0).to_numpy()
model_alert = scores >= threshold

out = {"split": args.split, "rows": len(y), "laundering": int(y.sum()), "threshold": threshold,
       "model_alerts": int(model_alert.sum()), "model_hits": int(y[model_alert].sum()), "rules": {}}
print(f"{args.split}: model alerts {out['model_alerts']:,}, laundering among them {out['model_hits']:,} "
      f"({out['model_hits'] / max(out['model_alerts'], 1):.1%}); all laundering {out['laundering']:,}")
for length in args.lengths:
    fires = cycle >= length
    extra = fires & ~model_alert
    row = {"fires": int(fires.sum()), "extra_alerts": int(extra.sum()), "extra_hits": int(y[extra].sum()),
           "fires_hits": int(y[fires].sum())}
    out["rules"][length] = row
    print(f"cycle >= {length}: fires {row['fires']:>5}, laundering {row['fires_hits']:>3} | extra alerts "
          f"{row['extra_alerts']:>5}, of which laundering {row['extra_hits']:>3} "
          f"({row['extra_hits'] / max(row['extra_alerts'], 1):.0%})")
(cfg.reports_dir / f"cycle_rule_{args.split}.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
