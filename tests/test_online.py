from datetime import datetime, timedelta

import pytest

from aegis.features import FEATURES, build_features
from aegis.online import FeatureEngine
from test_features import T0, make_tx


def online_rows(rows):
    engine = FeatureEngine({"US Dollar": 1.0}, set(), {})
    out = []
    for minutes, s, r, usd in rows:
        out.append(engine.process_ids(s, r, T0 + timedelta(minutes=minutes), usd, usd,
                                      "US Dollar", "US Dollar", "ACH"))
    return out


SCENARIOS = {
    "repeat_payer": [(0, 1, 2, 100.0), (10, 1, 3, 300.0), (20, 1, 2, 50.0)],
    "pass_through": [(0, 1, 2, 1000.0), (5, 2, 3, 990.0), (9, 3, 4, 970.0)],
    "window_expiry": [(0, 1, 2, 100.0), (60 * 25, 1, 3, 10.0), (60 * 26, 4, 1, 20.0)],
    "cycle": [(i * 600, i + 1, i + 2, 500.0) for i in range(5)] + [(3000, 6, 1, 480.0)],
    "self_transfer": [(0, 1, 1, 100.0), (10, 1, 2, 50.0), (20, 2, 1, 40.0)],
    "same_minute_window": [(0, 1, 2, 100.0), (0, 1, 3, 40.0), (0, 4, 2, 7.0), (1, 1, 2, 5.0)],
    "same_minute_lookup": [(0, 1, 2, 100.0), (30, 2, 3, 50.0), (31, 4, 2, 70.0), (31, 2, 5, 60.0)],
}


@pytest.mark.parametrize("name", SCENARIOS)
def test_online_features_match_batch_features(name):
    rows = SCENARIOS[name]
    batch = build_features(make_tx(rows)).select(FEATURES).to_dicts()
    for live, expected in zip(online_rows(rows), batch):
        for f in FEATURES:
            if expected[f] is None:
                assert live[f] is None, f
            else:
                assert live[f] == pytest.approx(expected[f], rel=1e-5, abs=1e-5), f


def test_new_accounts_never_reuse_an_existing_id():
    # gaps in the IDs, as left by trimming rows at ingest
    engine = FeatureEngine({"US Dollar": 1.0}, set(), {"1/A": 0, "1/B": 2, "1/C": 7})
    assert engine.account_id("1/A") == 0
    new = [engine.account_id(k) for k in ("2/X", "2/Y", "2/Z", "2/X")]
    assert new == [8, 9, 10, 8]


def test_timestamp_with_seconds_keeps_cycle_minutes():
    engine = FeatureEngine({"US Dollar": 1.0}, set(), {})
    when = datetime(2022, 9, 1, 8, 0, 30)
    values = engine.process_ids(1, 2, when, 10.0, 10.0, "US Dollar", "US Dollar", "ACH")
    assert values["hour"] == 8.0
