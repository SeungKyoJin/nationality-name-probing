"""
Loss 함수 정의 — auxiliary losses for name classification.

SupConLoss (Supervised Contrastive Loss):
  · 같은 label의 embedding은 가깝게, 다른 label은 멀게 학습
  · 논문 원본: Khosla et al. "Supervised Contrastive Learning" (NeurIPS 2020)

3가지 variant:
  · NAT:  nationality 기준 (168-class) — CNL 정렬 직접 유도
  · LANG: language group 기준 (38-class) — coarse signal, positive 많음
  · HIER: 두 loss 결합 (λ_lang × LANG + λ_nat × NAT) — 최종 제안

사용 예:
    from utils.losses import SupConLoss
    supcon = SupConLoss(temperature=0.07, variant='hier')
    loss = supcon(pooled_features, nat_labels, lang_labels)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SupConLoss(nn.Module):
    """
    Supervised Contrastive Loss.

    Args:
        temperature: softmax temperature (기본 0.07, 원본 논문 권장값)
        variant: 'nat' | 'lang' | 'hier'
        lang_weight: hier 모드에서 lang loss의 비중 (기본 0.3)
            → hier_loss = (1 - lang_weight) * nat_loss + lang_weight * lang_loss
    """

    def __init__(self, temperature: float = 0.07,
                 variant: str = 'hier',
                 lang_weight: float = 0.3):
        super().__init__()
        if variant not in ('nat', 'lang', 'hier'):
            raise ValueError(f"variant는 'nat'/'lang'/'hier' 중 하나: {variant}")
        self.temperature = temperature
        self.variant = variant
        self.lang_weight = lang_weight

    def forward(self, features: torch.Tensor,
                 nat_labels: torch.Tensor,
                 lang_labels: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            features:    [B, d] pooled embedding
            nat_labels:  [B]    nationality label
            lang_labels: [B]    language group label (hier/lang variant에만 필요)
        Returns:
            scalar loss
        """
        if self.variant == 'nat':
            return self._supcon_single(features, nat_labels)
        if self.variant == 'lang':
            if lang_labels is None:
                raise ValueError("lang variant는 lang_labels 필요")
            return self._supcon_single(features, lang_labels)
        # hier
        if lang_labels is None:
            raise ValueError("hier variant는 lang_labels 필요")
        loss_nat = self._supcon_single(features, nat_labels)
        loss_lang = self._supcon_single(features, lang_labels)
        return (1.0 - self.lang_weight) * loss_nat + self.lang_weight * loss_lang

    @staticmethod
    def _supcon_single(features: torch.Tensor,
                        labels: torch.Tensor,
                        temperature: float = 0.07) -> torch.Tensor:
        """
        단일 label 기준 SupCon (기본 구현).

        수식:
          L_i = -1/|P(i)| * Σ_{p ∈ P(i)} log(exp(z_i·z_p/τ) / Σ_{a ∈ A(i)} exp(z_i·z_a/τ))
        where
          · z = L2-normalized features
          · P(i) = same-label positives (excluding self)
          · A(i) = all except self
        """
        device = features.device
        # L2 normalize
        features = F.normalize(features, p=2, dim=1)

        # Similarity matrix (temperature scaled)
        logits = torch.matmul(features, features.T) / temperature        # [B, B]
        # 수치 안정성: 각 row 최대값 빼기
        logits_max, _ = logits.max(dim=1, keepdim=True)
        logits = logits - logits_max.detach()

        # Positive mask: 같은 label + self 제외
        labels = labels.contiguous().view(-1, 1)
        mask_same = torch.eq(labels, labels.T).float().to(device)        # [B, B]
        # self 제외
        B = features.size(0)
        self_mask = torch.eye(B, device=device)
        mask_pos = mask_same - self_mask
        mask_pos = mask_pos.clamp(min=0)                                  # [B, B]

        # Denominator: self 제외한 모든 샘플
        exp_logits = torch.exp(logits) * (1.0 - self_mask)
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-12)

        # Numerator: positive만 평균
        pos_counts = mask_pos.sum(dim=1)
        # positive가 없는 샘플은 loss=0 (batch에 same-label 없을 때)
        valid = pos_counts > 0
        if valid.sum() == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)

        mean_log_prob_pos = (mask_pos * log_prob).sum(dim=1)
        mean_log_prob_pos = mean_log_prob_pos[valid] / pos_counts[valid]

        return -mean_log_prob_pos.mean()
