from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.autograd import Function


class _GradientReversal(Function):
    @staticmethod
    def forward(ctx, inputs: torch.Tensor, scale: float) -> torch.Tensor:
        ctx.scale = scale
        return inputs.view_as(inputs)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return -ctx.scale * grad_output, None


def gradient_reversal(inputs: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    return _GradientReversal.apply(inputs, scale)


@dataclass
class ProxyBridgeOutput:
    latent_proxy: torch.Tensor
    reconstructed_proxy: torch.Tensor
    adversarial_logits: torch.Tensor


class ProxyBridgeMLP(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int, num_intervention_levels: int, depth: int = 2):
        super().__init__()
        hidden_dim = max(input_dim, latent_dim * 2)
        encoder_layers = []
        current_dim = input_dim
        for _ in range(max(depth, 1)):
            encoder_layers.extend([nn.Linear(current_dim, hidden_dim), nn.GELU()])
            current_dim = hidden_dim
        encoder_layers.append(nn.Linear(current_dim, latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, input_dim),
        )
        self.adversary = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_intervention_levels),
        )

    def forward(self, proxy_subspace: torch.Tensor, adversarial_scale: float = 1.0) -> ProxyBridgeOutput:
        latent_proxy = self.encoder(proxy_subspace)
        reconstructed_proxy = self.decoder(latent_proxy)
        adversarial_logits = self.adversary(gradient_reversal(proxy_subspace, scale=adversarial_scale))
        return ProxyBridgeOutput(
            latent_proxy=latent_proxy,
            reconstructed_proxy=reconstructed_proxy,
            adversarial_logits=adversarial_logits,
        )

