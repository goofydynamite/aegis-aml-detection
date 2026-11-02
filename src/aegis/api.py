"""Live scoring service.

python -m aegis state                                    # once, to build the account memory
uvicorn --factory aegis.api:load_app --port 8000         # then visit /docs

Every scored payment is added to the account history, so payments must arrive in time order.
History lives in this process's memory: it resets on restart and needs one worker.
"""
from __future__ import annotations

import json
import os
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import lightgbm as lgb
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from . import explain
from .config import Config
from .features import FEATURES
from .online import FeatureEngine, Payment

MAX_BATCH = 1000
DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "config.yaml"
AccountField = Field(min_length=1, max_length=64)
PaymentFormat = Literal["ACH", "Bitcoin", "Cash", "Cheque", "Credit Card", "Reinvestment", "Wire"]


class Transaction(BaseModel):
    model_config = ConfigDict(json_schema_extra={"example": {
        "timestamp": "2022-09-11T09:30:00", "from_bank": "011", "from_account": "80011B960",
        "to_bank": "000", "to_account": "80021EF10", "amount_paid": 11004.92,
        "payment_currency": "Euro", "receiving_currency": "Euro", "payment_format": "ACH",
    }})

    timestamp: datetime
    from_bank: str = AccountField
    from_account: str = AccountField
    to_bank: str = AccountField
    to_account: str = AccountField
    amount_paid: float = Field(gt=0, allow_inf_nan=False)
    payment_currency: str
    receiving_currency: str
    payment_format: PaymentFormat


class Score(BaseModel):
    risk_score: float
    alert: bool
    triggered_by: list[Literal["model", "cycle rule"]]
    reasons: list[str]
    cycle_length: int | None
    chain_depth: int


def _as_utc(when: datetime) -> datetime:
    return when if when.tzinfo is None else when.astimezone(timezone.utc).replace(tzinfo=None)


def create_app(engine: FeatureEngine, booster: lgb.Booster, threshold: float, dataset: str = "",
               cycle_alert_min_length: int | None = None) -> FastAPI:
    """cycle_alert_min_length: payments closing a cycle at least this long always alert."""
    app = FastAPI(title="Aegis", summary="Money-laundering risk scoring for payments")
    lock = threading.Lock()

    def check(tx: Transaction) -> None:
        for currency in (tx.payment_currency, tx.receiving_currency):
            if currency not in engine.rates:
                raise HTTPException(422, f"Unknown currency: {currency}")

    def score_one(tx: Transaction) -> Score:
        when = _as_utc(tx.timestamp)
        if engine.latest and when.replace(second=0, microsecond=0) < engine.latest.replace(second=0, microsecond=0):
            raise HTTPException(409, f"Payment is older than the latest one seen ({engine.latest.isoformat()})")
        values = engine.process(Payment(
            when, f"{tx.from_bank}/{tx.from_account}", f"{tx.to_bank}/{tx.to_account}", tx.amount_paid,
            tx.payment_currency, tx.receiving_currency, tx.payment_format,
        ))
        X = np.array([engine.vector(values)], dtype=np.float32)
        risk = float(booster.predict(X, num_threads=1)[0])
        cycle = None if values["cycle_length"] is None else int(values["cycle_length"])
        triggered = []
        if risk >= threshold:
            triggered.append("model")
        if cycle_alert_min_length and cycle is not None and cycle >= cycle_alert_min_length:
            triggered.append("cycle rule")
        return Score(
            risk_score=round(risk, 6),
            alert=bool(triggered),
            triggered_by=triggered,
            reasons=[r for r in explain.top_reasons(booster, X, FEATURES)[0].split("; ") if r],
            cycle_length=cycle,
            chain_depth=int(values["chain_depth"]),
        )

    @app.get("/health")
    def health() -> dict:
        return {
            "status": "ok", "dataset": dataset, "alert_threshold": threshold,
            "cycle_alert_min_length": cycle_alert_min_length,
            "accounts": len(engine.key_to_id), "transactions": engine.transactions,
            "latest_transaction": engine.latest,
        }

    @app.post("/score")
    def score(tx: Transaction) -> Score:
        check(tx)
        with lock:
            return score_one(tx)

    @app.post("/score/batch")
    def score_batch(transactions: list[Transaction]) -> list[Score]:
        if len(transactions) > MAX_BATCH:
            raise HTTPException(413, f"At most {MAX_BATCH} transactions per request")
        for tx in transactions:
            check(tx)
        ordered = sorted(enumerate(transactions), key=lambda item: _as_utc(item[1].timestamp))
        results: list[Score | None] = [None] * len(transactions)
        with lock:
            for i, tx in ordered:
                results[i] = score_one(tx)
        return results

    @app.get("/accounts/{bank}/{account}")
    def account_summary(bank: str, account: str) -> dict:
        account_id = engine.key_to_id.get(f"{bank}/{account}")
        history = engine.accounts.get(account_id) if account_id is not None else None
        if history is None:
            raise HTTPException(404, "Account not found")
        return {
            "payments_sent": history.out_count, "usd_sent": round(history.out_sum, 2),
            "accounts_paid": len(history.receivers), "payments_received": history.in_count,
            "usd_received": round(history.in_sum, 2), "accounts_paying_in": len(history.senders),
        }

    return app


def load_app() -> FastAPI:
    cfg = Config.load(os.environ.get("AEGIS_CONFIG", DEFAULT_CONFIG))
    booster = lgb.Booster(model_file=str(cfg.models_dir / f"{cfg.dataset}_lightgbm.txt"))
    started = time.time()
    engine = FeatureEngine.load(cfg.models_dir / f"{cfg.dataset}_state.pkl")
    meta = json.loads((cfg.models_dir / f"{cfg.dataset}_state.json").read_text(encoding="utf-8"))
    print(f"Loaded {len(engine.key_to_id):,} accounts in {time.time() - started:.0f}s", flush=True)
    rules = cfg.raw.get("rules", {})
    return create_app(engine, booster, meta["alert_threshold"], cfg.dataset,
                      rules.get("cycle_alert_min_length"))
