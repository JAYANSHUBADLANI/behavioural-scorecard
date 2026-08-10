"""Configuration loading and derived window parameters."""

from dataclasses import dataclass
from pathlib import Path
from typing import List

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "config.yaml"


@dataclass(frozen=True)
class WindowConfig:
    """Observation point and outcome window definition."""

    outcome_window_months: int
    dpd_threshold: int
    min_pre_history_months: int
    observation_step_months: int
    earliest_observation_month: int
    require_full_outcome_window: bool
    exclude_already_delinquent: bool
    require_active_limit_at_observation: bool = True
    recent_state_months: int = 3
    panel_last_month: int = -1

    @property
    def latest_observation_month(self) -> int:
        """Newest observation point that still leaves a full outcome window."""
        return self.panel_last_month - self.outcome_window_months

    def observation_months(self) -> List[int]:
        """Observation points from newest to oldest, spaced by the configured step."""
        if self.observation_step_months < 1:
            raise ValueError("observation_step_months must be at least 1")
        if self.latest_observation_month < self.earliest_observation_month:
            raise ValueError(
                f"outcome window of {self.outcome_window_months} months leaves no valid "
                f"observation point at or after {self.earliest_observation_month}"
            )
        return list(
            range(
                self.latest_observation_month,
                self.earliest_observation_month - 1,
                -self.observation_step_months,
            )
        )


@dataclass(frozen=True)
class TreatmentConfig:
    lookback_months: int
    min_increase_ratio: float


class Config:
    """Parsed project configuration with paths resolved against the project root."""

    def __init__(self, raw: dict, root: Path = PROJECT_ROOT):
        self._raw = raw
        self.root = root

    @classmethod
    def load(cls, path: Path = DEFAULT_CONFIG) -> "Config":
        with open(path) as handle:
            return cls(yaml.safe_load(handle))

    def path(self, key: str) -> Path:
        return (self.root / self._raw["paths"][key]).resolve()

    @property
    def window(self) -> WindowConfig:
        target = self._raw["target"]
        return WindowConfig(
            outcome_window_months=target["outcome_window_months"],
            dpd_threshold=target["dpd_threshold"],
            min_pre_history_months=target["min_pre_history_months"],
            observation_step_months=target["observation_step_months"],
            earliest_observation_month=target["earliest_observation_month"],
            require_full_outcome_window=target["require_full_outcome_window"],
            exclude_already_delinquent=target["exclude_already_delinquent"],
            require_active_limit_at_observation=target.get(
                "require_active_limit_at_observation", True
            ),
            recent_state_months=target.get("recent_state_months", 3),
            panel_last_month=self._raw["panel"]["last_month"],
        )

    @property
    def treatment(self) -> TreatmentConfig:
        block = self._raw["treatment"]
        return TreatmentConfig(
            lookback_months=block["lookback_months"],
            min_increase_ratio=block["min_increase_ratio"],
        )

    @property
    def panel_months(self) -> range:
        panel = self._raw["panel"]
        return range(panel["first_month"], panel["last_month"] + 1)

    def section(self, name: str) -> dict:
        return self._raw[name]
