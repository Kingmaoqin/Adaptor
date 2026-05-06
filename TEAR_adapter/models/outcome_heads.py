"""
Outcome Heads (修改 C)
======================
Treatment-specific outcome heads vs. factual single-head (ablation).

问题：
  旧版 outcome_head 接受 [z^C, z^Y, one_hot(k)] 的拼接，在全部 treatment levels
  上以 factual outcome loss 训练。这让 encoder 可能学到 treatment-conditional
  shortcut（factual T=1 路径与 T=0 路径各自有独立特征关联），
  导致反事实泛化能力受限。

修复：
  为每个 treatment level k 独立训练一个 head，仅使用 T_i=k 的样本训练 head_k：
      L_Y = (1/K) Σ_k MSE_k(head_k([z^C, z^Y]), Y)  where MSE_k is over T_i=k samples

  这样 head_k 不能利用 treatment indicator 作为 shortcut，只能从子空间表征预测结局。
"""
from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class TreatmentSpecificOutcomeHeads(nn.Module):
    """
    K independent outcome heads, one per treatment level.
    Each head_k is trained ONLY on samples where T_i = k.

    Input:  [z^C, z^Y]  (concatenated, shape (n, 2*d))
    Output: per-level outcome prediction  (n, K)
    """

    def __init__(self, embedding_dim: int, num_levels: int, hidden_dim: int = None):
        super().__init__()
        hidden_dim = hidden_dim or embedding_dim
        in_dim = embedding_dim * 2
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, 1),
            )
            for _ in range(num_levels)
        ])
        self.num_levels = num_levels

    def forward(self, z_conf: torch.Tensor, z_out: torch.Tensor) -> torch.Tensor:
        """
        Returns potential outcome predictions (n, K).
        All K heads are applied to all samples for inference.
        """
        inp = torch.cat([z_conf, z_out], dim=-1)   # (n, 2d)
        preds = [head(inp).squeeze(-1) for head in self.heads]  # K × (n,)
        return torch.stack(preds, dim=1)             # (n, K)

    def treatment_specific_loss(
        self,
        z_conf: torch.Tensor,
        z_out: torch.Tensor,
        Y: torch.Tensor,
        T: torch.Tensor,
    ) -> torch.Tensor:
        """
        L_Y = (1/K) Σ_k MSE(head_k([z^C, z^Y]), Y)  over T_i=k samples.
        Returns scalar loss.
        """
        inp = torch.cat([z_conf, z_out], dim=-1)  # (n, 2d)
        total_loss = torch.tensor(0.0, device=z_conf.device)
        n_heads_used = 0
        for k in range(self.num_levels):
            mask = T == k
            if mask.sum() == 0:
                continue
            pred_k = self.heads[k](inp[mask]).squeeze(-1)  # (n_k,)
            total_loss = total_loss + F.mse_loss(pred_k, Y[mask])
            n_heads_used += 1
        return total_loss / max(n_heads_used, 1)


class FactualSingleOutcomeHead(nn.Module):
    """
    Legacy single-head with treatment one-hot conditioning (factual).
    Kept for ablation comparison.

    Input:  [z^C, z^Y, one_hot(T)]   (shape (n, 2*d + K))
    Output: factual outcome prediction per potential level (n, K)
    """

    def __init__(self, embedding_dim: int, num_levels: int):
        super().__init__()
        in_dim = embedding_dim * 2 + num_levels
        self.head = nn.Sequential(
            nn.Linear(in_dim, embedding_dim),
            nn.GELU(),
            nn.Linear(embedding_dim, 1),
        )
        self.num_levels = num_levels

    def forward(self, z_conf: torch.Tensor, z_out: torch.Tensor) -> torch.Tensor:
        """Returns potential outcomes (n, K) by conditioning on each level k."""
        n = z_conf.shape[0]
        device = z_conf.device
        dtype = z_conf.dtype
        identity = torch.eye(self.num_levels, device=device, dtype=dtype)  # (K, K)
        preds = []
        for k in range(self.num_levels):
            oh_k = identity[k].unsqueeze(0).expand(n, -1)  # (n, K)
            inp_k = torch.cat([z_conf, z_out, oh_k], dim=-1)
            preds.append(self.head(inp_k).squeeze(-1))
        return torch.stack(preds, dim=1)  # (n, K)

    def factual_loss(
        self,
        z_conf: torch.Tensor,
        z_out: torch.Tensor,
        Y: torch.Tensor,
        T: torch.Tensor,
    ) -> torch.Tensor:
        """MSE on factual (observed) outcomes."""
        po_hat = self.forward(z_conf, z_out)  # (n, K)
        factual_pred = po_hat.gather(1, T.unsqueeze(-1)).squeeze(-1)
        return F.mse_loss(factual_pred, Y)
