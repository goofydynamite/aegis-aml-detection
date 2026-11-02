import lightgbm as lgb
import numpy as np
import pytest
from fastapi.testclient import TestClient

from aegis.api import MAX_BATCH, create_app
from aegis.features import FEATURES
from aegis.online import FeatureEngine


def tiny_booster() -> lgb.Booster:
    rng = np.random.default_rng(0)
    X = rng.random((400, len(FEATURES)))
    y = (X[:, 0] > 0.5).astype(int)
    return lgb.train({"objective": "binary", "verbosity": -1, "min_data_in_leaf": 5},
                     lgb.Dataset(X, y, feature_name=FEATURES), num_boost_round=5)


@pytest.fixture
def client():
    engine = FeatureEngine({"US Dollar": 1.0, "Euro": 0.85}, set(), {})
    return TestClient(create_app(engine, tiny_booster(), threshold=0.0, dataset="test"))


def payment(**overrides) -> dict:
    return {
        "timestamp": "2022-09-11T09:30:00", "from_bank": "1", "from_account": "A", "to_bank": "2",
        "to_account": "B", "amount_paid": 500.0, "payment_currency": "US Dollar",
        "receiving_currency": "US Dollar", "payment_format": "ACH", **overrides,
    }


def test_score_returns_risk_and_reasons(client):
    body = client.post("/score", json=payment()).json()
    assert 0 <= body["risk_score"] <= 1
    assert body["alert"] is True
    assert isinstance(body["reasons"], list)


def test_scoring_builds_account_history(client):
    client.post("/score", json=payment())
    client.post("/score", json=payment(timestamp="2022-09-11T09:40:00", amount_paid=100.0))
    body = client.get("/accounts/1/A").json()
    assert body["payments_sent"] == 2
    assert body["usd_sent"] == 600.0
    assert client.get("/health").json()["transactions"] == 2


def cycle_legs(client, legs):
    return [client.post("/score", json=payment(from_bank="1", from_account=s, to_bank="1", to_account=r,
                                               timestamp=f"2022-09-11T{t}:00")).json()
            for s, r, t in legs]


FOUR_HOP = [("A", "B", "09:00"), ("B", "C", "10:00"), ("C", "D", "11:00"), ("D", "A", "12:00")]


@pytest.fixture
def strict_client():
    """Model threshold too high to ever alert, so only the rule can."""
    engine = FeatureEngine({"US Dollar": 1.0}, set(), {})
    return TestClient(create_app(engine, tiny_booster(), threshold=2.0, cycle_alert_min_length=3))


def test_cycle_rule_alerts_on_a_long_cycle(strict_client):
    results = cycle_legs(strict_client, FOUR_HOP)
    assert [r["alert"] for r in results] == [False, False, False, True]
    assert results[-1]["triggered_by"] == ["cycle rule"]


def test_cycle_rule_ignores_short_cycles(strict_client):
    results = cycle_legs(strict_client, [("A", "B", "09:00"), ("B", "A", "10:00")])
    assert results[-1]["cycle_length"] == 2
    assert results[-1]["alert"] is False


def test_model_alert_names_the_model(client):
    body = client.post("/score", json=payment()).json()
    assert body["triggered_by"] == ["model"]


def test_cycle_is_reported_end_to_end(client):
    legs = [("A", "B", "09:00"), ("B", "C", "10:00"), ("C", "D", "11:00"), ("D", "A", "12:00")]
    results = [client.post("/score", json=payment(from_bank="1", from_account=s, to_bank="1", to_account=r,
                                                  timestamp=f"2022-09-11T{t}:00")).json()
               for s, r, t in legs]
    assert results[-1]["cycle_length"] == 4
    assert results[0]["cycle_length"] is None


def test_new_accounts_do_not_merge_with_existing_ones():
    engine = FeatureEngine({"US Dollar": 1.0}, set(), {"9/OLD1": 1, "9/OLD2": 3})
    client = TestClient(create_app(engine, tiny_booster(), threshold=0.0))
    client.post("/score", json=payment(from_bank="1", from_account="NEW", to_bank="2", to_account="NEW2"))
    assert client.get("/accounts/9/OLD1").status_code == 404
    assert client.get("/accounts/9/OLD2").status_code == 404
    assert client.get("/accounts/1/NEW").json()["payments_sent"] == 1


def test_older_payment_is_rejected(client):
    client.post("/score", json=payment())
    assert client.post("/score", json=payment(timestamp="2022-09-11T09:00:00")).status_code == 409


def test_same_minute_payments_are_accepted(client):
    client.post("/score", json=payment(timestamp="2022-09-11T09:30:40"))
    assert client.post("/score", json=payment(timestamp="2022-09-11T09:30:10")).status_code == 200


@pytest.mark.parametrize("bad", [
    {"amount_paid": -5}, {"amount_paid": 0}, {"payment_format": "Gold"},
    {"payment_currency": "Dogecoin"}, {"receiving_currency": "Dogecoin"}, {"from_account": ""},
])
def test_invalid_payment_is_rejected_and_not_recorded(client, bad):
    assert client.post("/score", json=payment(**bad)).status_code == 422
    assert client.get("/health").json()["transactions"] == 0


def test_timezone_offsets_are_converted_to_utc(client):
    client.post("/score", json=payment(timestamp="2022-09-11T09:30:00"))
    late_in_utc = payment(timestamp="2022-09-11T10:00:00+05:00")  # 05:00 UTC
    assert client.post("/score", json=late_in_utc).status_code == 409


def test_batch_keeps_request_order_and_scores_by_time(client):
    later = payment(timestamp="2022-09-11T10:00:00", amount_paid=1.0)
    earlier = payment(timestamp="2022-09-11T09:00:00", amount_paid=2.0)
    response = client.post("/score/batch", json=[later, earlier])
    assert response.status_code == 200
    assert len(response.json()) == 2
    assert client.get("/accounts/1/A").json()["payments_sent"] == 2


def test_batch_with_a_bad_payment_records_nothing(client):
    response = client.post("/score/batch", json=[payment(), payment(payment_currency="Dogecoin")])
    assert response.status_code == 422
    assert client.get("/health").json()["transactions"] == 0


def test_oversized_batch_is_rejected(client):
    assert client.post("/score/batch", json=[payment()] * (MAX_BATCH + 1)).status_code == 413


def test_unknown_account_is_404(client):
    assert client.get("/accounts/9/Z").status_code == 404
