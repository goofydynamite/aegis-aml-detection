"""Configuration loading and path resolution."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class Config:
    root: Path
    raw: dict[str, Any]

    @classmethod
    def load(cls, path: str | Path = "config.yaml") -> "Config":
        path = Path(path).resolve()
        with path.open(encoding="utf-8") as f:
            return cls(root=path.parent, raw=yaml.safe_load(f))

    def __getitem__(self, key: str) -> Any:
        return self.raw[key]

    @property
    def seed(self) -> int:
        return int(self.raw["seed"])

    @property
    def dataset(self) -> str:
        return self.raw["data"]["dataset"]

    def _dir(self, section: str, key: str) -> Path:
        d = self.root / self.raw[section][key]
        d.mkdir(parents=True, exist_ok=True)
        return d

    @property
    def raw_transactions(self) -> Path:
        return self.root / self.raw["data"]["raw_dir"] / f"{self.dataset}_Trans.csv"

    @property
    def raw_patterns(self) -> Path:
        return self.root / self.raw["data"]["raw_dir"] / f"{self.dataset}_Patterns.txt"

    @property
    def transactions_parquet(self) -> Path:
        return self._dir("data", "processed_dir") / f"{self.dataset}_transactions.parquet"

    @property
    def features_parquet(self) -> Path:
        return self._dir("data", "processed_dir") / f"{self.dataset}_features.parquet"

    @property
    def models_dir(self) -> Path:
        return self._dir("outputs", "models_dir")

    @property
    def reports_dir(self) -> Path:
        return self._dir("outputs", "reports_dir")
