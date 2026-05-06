from __future__ import annotations

from dataclasses import dataclass


@dataclass
class WarmupSchedule:
    total_epochs: int
    warmup_fraction: float

    @property
    def num_warmup_epochs(self) -> int:
        return max(1, int(self.total_epochs * self.warmup_fraction))

