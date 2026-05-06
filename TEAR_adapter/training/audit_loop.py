from __future__ import annotations

import math


class DecayingAuditScheduler:
    def __init__(self, base_weight: float, num_features: int, num_samples: int, total_epochs: int):
        self.base_weight = base_weight
        self.high_dimensional_scale = min(1.0, math.sqrt(math.log(max(num_features, 2)) / max(num_samples, 1)))
        self.total_epochs = total_epochs

    def weight(self, epoch_index: int, confidence: float) -> float:
        return float(
            self.base_weight
            * confidence
            * self.high_dimensional_scale
            * math.exp(-float(epoch_index) / max(float(self.total_epochs), 1.0))
        )

