"""Live feature engine: the same features as features.py, one transaction at a time.

The batch pipeline reads a whole file; a live payment has only what came before it, so
this keeps each account's running history in memory. scripts/check_online_parity.py
compares its output with the training features on real data.
"""
from __future__ import annotations

import math
import pickle
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import polars as pl

from .cycles import CycleTracker
from .data import PAYMENT_FORMATS
from .features import FEATURES
from .graph import hub_accounts

DAY_SECS = 86_400
_FORMAT_CODES = {name: i for i, name in enumerate(PAYMENT_FORMATS)}


class Account:
    """Running history of one account."""

    __slots__ = ("out_count", "out_sum", "last_out", "out_window", "out_window_sum", "receivers",
                 "out_snaps", "in_count", "in_sum", "last_in", "in_window", "in_window_sum",
                 "senders", "in_snaps")

    def __init__(self) -> None:
        self.out_count = 0
        self.out_sum = 0.0
        self.last_out: int | None = None
        self.out_window: deque = deque()  # (ts, usd) inside the last 24h
        self.out_window_sum = 0.0
        self.receivers: set[int] = set()
        self.out_snaps: list[tuple] = []  # newest per minute, two kept
        self.in_count = 0
        self.in_sum = 0.0
        self.last_in: int | None = None
        self.in_window: deque = deque()
        self.in_window_sum = 0.0
        self.senders: set[int] = set()
        self.in_snaps: list[tuple] = []


@dataclass
class Payment:
    timestamp: datetime
    from_key: str
    to_key: str
    amount_paid: float
    payment_currency: str
    receiving_currency: str
    payment_format: str


def _trim(window: deque, total: float, now: int) -> float:
    """Drop entries older than 24h and return the running sum of what is left.

    A running sum keeps busy hub accounts as fast as quiet ones.
    """
    cutoff = now - DAY_SECS
    while window and window[0][0] <= cutoff:
        total -= window.popleft()[1]
    return total if window else 0.0


def _before(snaps: list[tuple], ts: int) -> tuple | None:
    """Latest snapshot from an earlier minute."""
    for snap in reversed(snaps):
        if snap[0] < ts:
            return snap
    return None


def _push(snaps: list[tuple], snap: tuple) -> None:
    if snaps and snaps[-1][0] == snap[0]:
        snaps[-1] = snap
    else:
        snaps.append(snap)
        del snaps[:-2]


def _ratio(top: float, bottom: float | None) -> float | None:
    if bottom is None or bottom == 0:
        return None
    return top / bottom


class FeatureEngine:
    def __init__(self, rates: dict[str, float], hubs: set[int], key_to_id: dict[str, int]):
        self.rates = rates
        self.key_to_id = key_to_id
        self.next_id = max(key_to_id.values(), default=-1) + 1  # IDs can have gaps
        self.accounts: dict[int, Account] = {}
        self.cycles = CycleTracker(hubs)
        self.transactions = 0
        self.latest: datetime | None = None

    def account_id(self, key: str) -> int:
        account_id = self.key_to_id.get(key)
        if account_id is None:
            account_id = self.key_to_id[key] = self.next_id
            self.next_id += 1
        return account_id

    def _account(self, account_id: int) -> Account:
        account = self.accounts.get(account_id)
        if account is None:
            account = self.accounts[account_id] = Account()
        return account

    def to_usd(self, amount: float, currency: str) -> float:
        return amount / self.rates[currency]

    def process(self, p: Payment) -> dict[str, float | None]:
        s_id, r_id = self.account_id(p.from_key), self.account_id(p.to_key)
        return self.process_ids(s_id, r_id, p.timestamp, p.amount_paid, self.to_usd(p.amount_paid, p.payment_currency),
                                 p.payment_currency, p.receiving_currency, p.payment_format)

    def process_ids(self, s_id: int, r_id: int, when: datetime, amount_paid: float, usd: float,
                    pay_currency: str, recv_currency: str, fmt: str) -> dict[str, float | None]:
        """Features for one payment, then add the payment to the account history."""
        ts = int(when.replace(tzinfo=timezone.utc).timestamp())
        s, r = self._account(s_id), self._account(r_id)

        s.out_window_sum = _trim(s.out_window, s.out_window_sum, ts)
        r.in_window_sum = _trim(r.in_window, r.in_window_sum, ts)
        out_n, in_n = len(s.out_window), len(r.in_window)
        out_sum, in_sum = s.out_window_sum, r.in_window_sum
        is_new_receiver = r_id not in s.receivers
        is_new_sender = s_id not in r.senders
        distinct_receivers = len(s.receivers) + is_new_receiver
        distinct_senders = len(r.senders) + is_new_sender
        last_in = _before(s.in_snaps, ts)
        last_out = _before(r.out_snaps, ts)
        length, minutes, depth, sources = self.cycles.observe(s_id, r_id, ts // 60)

        values = {
            "log_amount_usd": math.log1p(usd),
            "is_cross_currency": float(pay_currency != recv_currency),
            "is_self_transfer": float(s_id == r_id),
            "is_round_amount": float(amount_paid % 100 == 0),
            "hour": float(when.hour),
            "weekday": float(when.isoweekday()),
            "payment_format_code": float(_FORMAT_CODES[fmt]),
            "s_out_prior_count": float(s.out_count),
            "s_out_count_24h": float(out_n + 1),
            "s_out_sum_24h": out_sum + usd,
            "s_distinct_receivers": float(distinct_receivers),
            "s_secs_since_prev_out": None if s.last_out is None else float(ts - s.last_out),
            "amount_vs_sender_mean": _ratio(usd, s.out_sum / s.out_count) if s.out_count else None,
            "s_is_new_receiver": float(is_new_receiver),
            "r_in_prior_count": float(r.in_count),
            "r_in_count_24h": float(in_n + 1),
            "r_in_sum_24h": in_sum + usd,
            "r_distinct_senders": float(distinct_senders),
            "r_secs_since_prev_in": None if r.last_in is None else float(ts - r.last_in),
            "amount_vs_receiver_mean": _ratio(usd, r.in_sum / r.in_count) if r.in_count else None,
            "s_secs_since_last_in": float(ts - last_in[0]) if last_in else None,
            "s_last_in_amount_ratio": _ratio(usd, last_in[1]) if last_in else None,
            "s_in_distinct_senders": float(last_in[2]) if last_in else None,
            "s_in_count_24h_at_last_in": float(last_in[3]) if last_in else None,
            "r_out_distinct_receivers": float(last_out[1]) if last_out else None,
            "r_out_prior_count": float(last_out[2]) if last_out else None,
            "cycle_length": None if length is None else float(length),
            "cycle_secs": None if minutes is None else float(minutes * 60),
            "chain_depth": float(depth),
            "chain_sources": float(sources),
        }

        s.out_window.append((ts, usd))
        s.out_window_sum += usd
        s.out_sum += usd
        s.receivers.add(r_id)
        _push(s.out_snaps, (ts, distinct_receivers, s.out_count + 1))
        s.out_count += 1
        s.last_out = ts
        r.in_window.append((ts, usd))
        r.in_window_sum += usd
        r.in_sum += usd
        r.senders.add(s_id)
        _push(r.in_snaps, (ts, usd, distinct_senders, in_n + 1))
        r.in_count += 1
        r.last_in = ts
        self.transactions += 1
        self.latest = when
        return values

    def vector(self, values: dict[str, float | None]) -> list[float]:
        return [math.nan if values[f] is None else values[f] for f in FEATURES]

    def save(self, path: str | Path) -> None:
        with open(path, "wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load(path: str | Path) -> "FeatureEngine":
        with open(path, "rb") as f:
            return pickle.load(f)


REPLAY_COLUMNS = ["sender", "receiver", "timestamp", "amount_paid", "amount_usd", "payment_currency",
                  "receiving_currency", "payment_format", "from_bank", "from_account", "to_bank", "to_account"]


def new_engine(tx: pl.DataFrame, hub_cap: int = 1000) -> FeatureEngine:
    """Empty engine using the exchange rates, hubs and account IDs found in tx."""
    rates = dict(
        tx.filter(pl.col("amount_usd") > 0)
        .group_by("payment_currency")
        .agg((pl.col("amount_paid") / pl.col("amount_usd")).median().alias("rate"))
        .iter_rows()
    )
    keys = pl.concat([
        tx.select(key=pl.col("from_bank") + "/" + pl.col("from_account"), id="sender"),
        tx.select(key=pl.col("to_bank") + "/" + pl.col("to_account"), id="receiver"),
    ]).unique()
    key_to_id = dict(zip(keys["key"].to_list(), keys["id"].to_list()))
    return FeatureEngine(rates, set(hub_accounts(tx, hub_cap)["account"].to_list()), key_to_id)


def replay_rows(tx: pl.DataFrame):
    """Rows of tx as process_ids arguments."""
    return zip(tx["sender"].to_list(), tx["receiver"].to_list(), tx["timestamp"].to_list(),
               tx["amount_paid"].to_list(), tx["amount_usd"].to_list(), tx["payment_currency"].to_list(),
               tx["receiving_currency"].to_list(), tx["payment_format"].cast(pl.String).to_list())
