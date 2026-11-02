from datetime import datetime, timedelta

import polars as pl

from aegis.data import trim_sparse_tail


def test_trim_sparse_tail_drops_near_empty_final_days():
    stamps = [datetime(2022, 9, d, 12) + timedelta(minutes=i) for d in (1, 2, 3) for i in range(200)]
    stamps += [datetime(2022, 9, 4, 9), datetime(2022, 9, 5, 9)]  # sparse tail
    out = trim_sparse_tail(pl.DataFrame({"timestamp": stamps}), min_day_share=0.01)
    assert out["timestamp"].max().day == 3
    assert len(out) == 600
