"""Chronological splitting and the detection models."""
from __future__ import annotations

import lightgbm as lgb
import numpy as np
import polars as pl
from sklearn.ensemble import IsolationForest

from .features import CATEGORICAL_FEATURES


def time_split(df: pl.DataFrame, train_frac: float, valid_frac: float) -> tuple[pl.DataFrame, ...]:
    """Split a time-sorted frame into train/valid/test on minute boundaries."""
    ts = df["timestamp"]
    n = len(df)
    t_valid = ts[int(n * train_frac)]
    t_test = ts[int(n * (train_frac + valid_frac))]
    return (
        df.filter(pl.col("timestamp") < t_valid),
        df.filter((pl.col("timestamp") >= t_valid) & (pl.col("timestamp") < t_test)),
        df.filter(pl.col("timestamp") >= t_test),
    )


def to_matrix(df: pl.DataFrame, features: list[str]) -> np.ndarray:
    return df.select(pl.col(features).cast(pl.Float32)).to_numpy()


def train_lightgbm(
    X_train: np.ndarray, y_train: np.ndarray,
    X_valid: np.ndarray, y_valid: np.ndarray,
    features: list[str], params: dict, seed: int,
) -> lgb.Booster:
    params = dict(params)
    num_boost_round = params.pop("num_boost_round")
    early_stopping_rounds = params.pop("early_stopping_rounds")
    params.update(objective="binary", metric="average_precision", seed=seed,
                  deterministic=True, verbosity=-1)
    cats = [f for f in CATEGORICAL_FEATURES if f in features]
    train_set = lgb.Dataset(X_train, y_train, feature_name=features, categorical_feature=cats)
    valid_set = lgb.Dataset(X_valid, y_valid, reference=train_set)
    callbacks = [lgb.log_evaluation(100)]
    if early_stopping_rounds:
        callbacks.append(lgb.early_stopping(early_stopping_rounds, verbose=False))
    return lgb.train(params, train_set, num_boost_round=num_boost_round, valid_sets=[valid_set],
                     callbacks=callbacks)


def train_isolation_forest(X_train: np.ndarray, n_estimators: int, fit_sample_size: int,
                           seed: int) -> IsolationForest:
    """Unsupervised baseline: fit on a sample of training data, labels unused."""
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X_train), size=min(fit_sample_size, len(X_train)), replace=False)
    model = IsolationForest(n_estimators=n_estimators, random_state=seed, n_jobs=-1)
    model.fit(np.nan_to_num(X_train[idx], nan=-1.0))
    return model


def isolation_scores(model: IsolationForest, X: np.ndarray) -> np.ndarray:
    # score_samples is higher for normal points; negate so higher means riskier
    return -model.score_samples(np.nan_to_num(X, nan=-1.0))
