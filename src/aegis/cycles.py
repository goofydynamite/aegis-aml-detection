"""Cycle and chain detection of any length, by tracking where money could have come from.

Payments are seen in time order. Each account keeps the accounts that could have funded
it through a time-respecting path in the last WINDOW_MINUTES, with the path length. A
payment back to one of those accounts closes a cycle. A minute's payments only take
effect once the next minute starts, so nothing depends on later data.
"""
from __future__ import annotations

import polars as pl

from .graph import hub_accounts

WINDOW_MINUTES = 96 * 60  # the slowest cycles in AMLworld take about 95 hours
MAX_SOURCES = 32          # sources remembered per account; the oldest is dropped first
CYCLE_FEATURES = ["cycle_length", "cycle_secs", "chain_depth", "chain_sources"]


class CycleTracker:
    """Shared by the batch replay and the live scoring service."""

    def __init__(self, skip: set[int], window: int = WINDOW_MINUTES, max_sources: int = MAX_SOURCES):
        self.skip = skip
        self.window = window
        self.max_sources = max_sources
        self.sources: dict[int, dict[int, tuple[int, int]]] = {}  # account -> {source: (start, hops)}
        self.pending: list[tuple[int, int, int, dict]] = []
        self.current: int | None = None

    def _apply_pending(self) -> None:
        for s, r, now, known in self.pending:
            target = self.sources.setdefault(r, {})
            target[s] = (now, 1)
            for src, (start, hops) in known.items():
                if src != r and (src not in target or target[src][0] < start):
                    target[src] = (start, hops + 1)
            if len(target) > self.max_sources:
                oldest = sorted(target, key=lambda k: target[k][0])
                for src in oldest[:len(target) - self.max_sources]:
                    del target[src]
        self.pending.clear()

    def observe(self, s: int, r: int, now: int) -> tuple[int | None, int | None, int, int]:
        """Record a payment; return (cycle_length, cycle_minutes, chain_depth, chain_sources)."""
        if now != self.current:
            self._apply_pending()
            self.current = now
        if s == r or s in self.skip or r in self.skip:
            return None, None, 0, 0

        length, minutes, depth, count = None, None, 0, 0
        known = self.sources.get(s)
        if known:
            cutoff = now - self.window
            for src in [k for k, (start, _) in known.items() if start < cutoff]:
                del known[src]
            if known:
                count = len(known)
                depth = max(hops for _, hops in known.values())
                if r in known:
                    start, hops = known[r]
                    length, minutes = hops + 1, now - start
        self.pending.append((s, r, now, dict(known) if known else {}))
        return length, minutes, depth, count


def replay(senders: list[int], receivers: list[int], minutes: list[int], skip: set[int],
           window: int = WINDOW_MINUTES, max_sources: int = MAX_SOURCES):
    """Return (cycle_length, cycle_minutes, chain_depth, chain_sources), one value per payment."""
    tracker = CycleTracker(skip, window, max_sources)
    return tuple(map(list, zip(*(
        tracker.observe(s, r, now) for s, r, now in zip(senders, receivers, minutes)
    )))) if senders else ([], [], [], [])


def add_cycle_features(df: pl.DataFrame, hub_cap: int = 1000) -> pl.DataFrame:
    """Expects df sorted by timestamp, with sender, receiver and timestamp columns."""
    hubs = set(hub_accounts(df, hub_cap)["account"].to_list())
    minutes = (df["timestamp"].dt.epoch("s") // 60).to_list()
    length, mins, depth, count = replay(df["sender"].to_list(), df["receiver"].to_list(), minutes, hubs)
    return df.with_columns(
        cycle_length=pl.Series(length, dtype=pl.Float32),
        cycle_secs=pl.Series(mins, dtype=pl.Float32) * 60,
        chain_depth=pl.Series(depth, dtype=pl.Float32),
        chain_sources=pl.Series(count, dtype=pl.Float32),
    )
