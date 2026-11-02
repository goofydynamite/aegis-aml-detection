from datetime import datetime, timedelta

import polars as pl
import pytest

from aegis.data import PAYMENT_FORMAT_DTYPE
from aegis.features import FEATURES, build_features
from aegis.graph import GRAPH_FEATURES

T0 = datetime(2022, 9, 1, 8, 0)


def make_tx(rows: list[tuple[int, int, int, float]]) -> pl.DataFrame:
    """rows: (minutes after T0, sender, receiver, amount_usd)."""
    return pl.DataFrame({
        "txn_id": list(range(len(rows))),
        "timestamp": [T0 + timedelta(minutes=m) for m, *_ in rows],
        "sender": [s for _, s, _, _ in rows],
        "receiver": [r for _, _, r, _ in rows],
        "amount_paid": [a for *_, a in rows],
        "amount_usd": [a for *_, a in rows],
        "payment_currency": ["US Dollar"] * len(rows),
        "receiving_currency": ["US Dollar"] * len(rows),
        "payment_format": pl.Series(["ACH"] * len(rows), dtype=PAYMENT_FORMAT_DTYPE),
        "is_laundering": [0] * len(rows),
        "pattern": pl.Series([None] * len(rows), dtype=pl.String),
    }).with_columns(pl.col("sender", "receiver").cast(pl.UInt32))


def test_sender_history_counts_only_earlier_payments():
    f = build_features(make_tx([(0, 1, 2, 100.0), (10, 1, 3, 300.0), (20, 1, 2, 50.0)]))
    assert f["s_out_prior_count"].to_list() == [0, 1, 2]
    assert f["s_distinct_receivers"].to_list() == [1, 2, 2]
    assert f["s_is_new_receiver"].to_list() == [1, 1, 0]
    assert f["amount_vs_sender_mean"][2] == pytest.approx(50.0 / 200.0)
    assert f["amount_vs_sender_mean"][0] is None


def test_24h_window_drops_old_payments():
    f = build_features(make_tx([(0, 1, 2, 100.0), (60 * 25, 1, 3, 10.0)]))
    assert f["s_out_count_24h"].to_list() == [1, 1]
    assert f["s_out_sum_24h"][1] == pytest.approx(10.0)


def test_window_does_not_see_later_payments_in_the_same_minute():
    f = build_features(make_tx([(0, 1, 2, 100.0), (0, 1, 3, 40.0), (0, 4, 2, 7.0)]))
    assert f["s_out_count_24h"].to_list() == [1, 2, 1]
    assert f["s_out_sum_24h"].to_list() == pytest.approx([100.0, 140.0, 7.0])
    assert f["r_in_count_24h"].to_list() == [1, 1, 2]


def test_pass_through_is_visible_from_sender_side():
    # account 2 receives 1000 and forwards 990 five minutes later
    f = build_features(make_tx([(0, 1, 2, 1000.0), (5, 2, 3, 990.0)]))
    assert f["s_secs_since_last_in"][1] == pytest.approx(300.0)
    assert f["s_last_in_amount_ratio"][1] == pytest.approx(0.99)
    assert f["s_secs_since_last_in"][0] is None


def test_fan_in_counts_distinct_senders():
    f = build_features(make_tx([(i, 10 + i, 99, 500.0) for i in range(5)] + [(9, 10, 99, 1.0)]))
    assert f["r_distinct_senders"].to_list() == [1, 2, 3, 4, 5, 5]


def test_features_do_not_look_ahead():
    base = [(0, 1, 2, 100.0), (30, 2, 3, 90.0), (60, 1, 3, 40.0)]
    future = [(120, 3, 1, 5000.0), (130, 2, 1, 7000.0), (140, 1, 2, 1.0)]
    a = build_features(make_tx(base)).select(FEATURES + GRAPH_FEATURES)
    b = build_features(make_tx(base + future)).select(FEATURES + GRAPH_FEATURES).head(len(base))
    assert a.equals(b)


def test_long_cycle_is_closed_by_the_final_payment():
    # 1 -> 2 -> 3 -> 4 -> 5 -> 6 -> 1 with amounts that drift
    hops = [(i * 600, i + 1, i + 2, 1000.0 * (1 + i % 3)) for i in range(5)]
    f = build_features(make_tx(hops + [(3000, 6, 1, 777.0)]))
    assert f["cycle_length"][5] == 6
    assert f["cycle_secs"][5] == pytest.approx(3000 * 60)
    assert f["cycle_length"].head(5).null_count() == 5
    assert f["chain_depth"].to_list() == [0, 1, 2, 3, 4, 5]


def test_cycle_expires_outside_the_window():
    f = build_features(make_tx([(0, 1, 2, 10.0), (100, 2, 3, 10.0), (97 * 60, 3, 1, 10.0)]))
    assert f["cycle_length"][2] is None


def test_path_must_respect_time_order():
    # 2 -> 3 happens before 1 -> 2, so nothing paid by 1 can have reached 3
    f = build_features(make_tx([(0, 2, 3, 10.0), (10, 1, 2, 10.0), (20, 3, 1, 10.0)]))
    assert f["cycle_length"][2] is None


def test_reverse_payment_is_found():
    f = build_features(make_tx([(0, 1, 2, 500.0), (30, 2, 1, 490.0)]))
    assert f["secs_since_reverse_payment"][1] == pytest.approx(1800.0)


def test_parallel_paths_count_distinct_routes_into_a_target():
    # 1 scatters to 2, 3, 4 and all three gather into 9
    scatter = [(0, 1, m, 100.0) for m in (2, 3, 4)]
    gather = [(10 * i, m, 9, 95.0) for i, m in enumerate((2, 3, 4), start=1)]
    later = [(60, 5, 9, 10.0)]
    f = build_features(make_tx(scatter + gather + later))
    assert f["r_max_parallel_paths"][-1] == 3
