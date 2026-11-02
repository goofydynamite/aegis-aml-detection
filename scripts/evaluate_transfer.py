"""Score one dataset with a model trained on another, without retraining.

Usage: python scripts/evaluate_transfer.py --config config.hi-small.yaml --model models/LI-Small_lightgbm.txt
"""
from __future__ import annotations

import argparse
import json

import lightgbm as lgb
import polars as pl

from aegis import evaluate, features, model
from aegis.config import Config

parser = argparse.ArgumentParser()
parser.add_argument("--config", required=True)
parser.add_argument("--model", required=True)
args = parser.parse_args()

cfg = Config.load(args.config)
ev = cfg["evaluation"]
feats = pl.read_parquet(cfg.features_parquet, columns=[*features.FEATURES, "timestamp", "is_laundering", "pattern"])
_, _, test = model.time_split(feats, **cfg["split"])
del feats

booster = lgb.Booster(model_file=args.model)
scores = booster.predict(model.to_matrix(test, features.FEATURES))
y = test["is_laundering"].to_numpy()
metrics = evaluate.evaluate_scores(y, scores, ev["k_values"])
metrics["pattern_recall"] = evaluate.pattern_recall(test["pattern"].to_numpy(), scores, ev["pattern_recall_k"])
metrics["model"] = args.model

(cfg.reports_dir / "transfer_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
print(json.dumps({k: v for k, v in metrics.items() if k != "pattern_recall"}, indent=2))
