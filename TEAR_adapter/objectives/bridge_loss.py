from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from ..models.proxy_bridge import ProxyBridgeOutput


class ProxyBridgeBottleneckLoss(nn.Module):
    def __init__(self, keep_weight: float = 1.0, leak_weight: float = 0.1):
        super().__init__()
        self.keep_weight = keep_weight
        self.leak_weight = leak_weight

    def forward(
        self,
        bridge_output: ProxyBridgeOutput,
        proxy_subspace: torch.Tensor,
        intervention: torch.Tensor,
    ) -> torch.Tensor:
        keep_term = F.mse_loss(bridge_output.reconstructed_proxy, proxy_subspace.detach())
        leak_term = F.cross_entropy(bridge_output.adversarial_logits, intervention)
        return self.keep_weight * keep_term + self.leak_weight * leak_term

