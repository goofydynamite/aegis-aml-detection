"""Compare feature sets at a fixed tree count over several seeds, on validation data only.

Early stopping lands on very different tree counts between runs, which makes single-run
comparisons noisy. The test set stays out of this, so it can still give an honest final score.

Usage: python scripts/compare_feature_sets.py [--rounds 400] [--seeds 1 2]
"""
from __future__ import annotations

import argparse
import json
import statistics

import lightgbm as lgb
import polars as pl

from aegis import evaluate, features, model
from aegis.config import Config
from aegis.cycles import CYCLE_FEATURES
from aegis.graph import GRAPH_FEATURES

parser = argparse.ArgumentParser()
parser.add_argument("--rounds", type=int, default=400)
parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2])
args = parser.parse_args()

cfg = Config.load("config.yaml")
feats = pl.read_parquet(cfg.features_parquet)
train, valid, _ = model.time_split(feats, **cfg["split"])
del feats
y_train, y_valid = train["is_laundering"].to_numpy(), valid["is_laundering"].to_numpy()

behavioural = [f for f in features.FEATURES if f not in CYCLE_FEATURES]
sets = {
    "behavioural": behavioural,
    "+paths": behavioural + GRAPH_FEATURES,
    "+cycles": behavioural + CYCLE_FEATURES,
    "+paths+cycles": behavioural + GRAPH_FEATURES + CYCLE_FEATURES,
}
params = {k: v for k, v in cfg["lightgbm"].items() if k not in ("num_boost_round", "early_stopping_rounds")}
params.update(objective="binary", verbosity=-1, deterministic=True)

results = {}
for name, cols in sets.items():
    X_train, X_valid = model.to_matrix(train, cols), model.to_matrix(valid, cols)
    runs = []
    for seed in args.seeds:
        data = lgb.Dataset(X_train, y_train, feature_name=cols,
                           categorical_feature=[c for c in features.CATEGORICAL_FEATURES if c in cols])
        booster = lgb.train({**params, "seed": seed}, data, num_boost_round=args.rounds)
        runs.append(evaluate.evaluate_scores(y_valid, booster.predict(X_valid), [100, 1000, 5000]))
        r = runs[-1]
        print(f"{name:<14} seed {seed}  PR-AUC {r['pr_auc']:.4f}  R@1000 {r['recall@1000']:.3f}  "
              f"R@5000 {r['recall@5000']:.3f}", flush=True)
    results[name] = {
        k: statistics.mean(r[k] for r in runs) for k in ("pr_auc", "recall@1000", "recall@5000")
    }
    results[name]["runs"] = [r["pr_auc"] for r in runs]
    del X_train, X_valid

(cfg.reports_dir / "feature_set_comparison.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
for name, r in results.items():
    print(f"{name:<14} mean PR-AUC {r['pr_auc']:.4f}  R@1000 {r['recall@1000']:.3f}  R@5000 {r['recall@5000']:.3f}")
