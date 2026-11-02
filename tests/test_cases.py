from datetime import datetime

import polars as pl

from aegis.cases import account_risk, assign_cases, summarise_cases


def alerts() -> pl.DataFrame:
    return pl.DataFrame({
        "from_bank": ["1", "1", "2", "7"],
        "from_account": ["A", "B", "X", "M"],
        "to_bank": ["1", "1", "2", "7"],
        "to_account": ["B", "C", "Y", "N"],
        "amount_usd": [100.0, 90.0, 50.0, 10.0],
        "risk_score": [0.9, 0.8, 0.7, 0.95],
        "is_laundering": [1, 1, 0, 0],
        "timestamp": [datetime(2022, 9, 9, h) for h in (1, 2, 3, 4)],
    })


def test_alerts_sharing_an_account_form_one_case():
    a = assign_cases(alerts())
    assert a.filter(pl.col("from_account") == "A")["case_id"][0] == 1
    assert a.filter(pl.col("from_account") == "B")["case_id"][0] == 1
    assert a["case_id"].n_unique() == 3


def test_case_summary_and_account_rollup():
    a = assign_cases(alerts())
    s = summarise_cases(a).filter(pl.col("case_id") == 1)
    assert s["alerts"][0] == 2
    assert s["accounts"][0] == 3
    assert s["total_usd"][0] == 190.0
    r = account_risk(a).filter(pl.col("account") == "1/B")
    assert r["alerts"][0] == 2 and r["as_sender"][0] == 1 and r["as_receiver"][0] == 1
