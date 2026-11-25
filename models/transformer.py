import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    """
    Single-channel Transformer Encoder.
    출력: (nat_logits, lang_logits)
      - nat_logits:  [batch, n_classes]
      - lang_logits: [batch, n_lang_groups]
    """

    def __init__(self, n_classes: int, n_lang_groups: int, d_model: int, n_head: int,
                 d_hid: int, n_layers: int, vocab_size, pad_token_id=0, dropout: float = 0.5):
        super().__init__()
        self.d_model = d_model
        self.pad_token_id = pad_token_id

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model, n_head, d_hid, dropout, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, n_layers)
        self.pooling = AttentionPooling(d_model)

        # Dual head
        self.head_nat = nn.Linear(d_model, n_classes)
        self.head_lang = nn.Linear(d_model, n_lang_groups)

    def forward(self, x):
        padding_mask = (x == self.pad_token_id)
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x, src_key_padding_mask=padding_mask)
        pooled = self.pooling(x, mask=padding_mask)

        nat_logits = self.head_nat(pooled)
        lang_logits = self.head_lang(pooled)
        return nat_logits, lang_logits


class GatedFusion(nn.Module):
    """
    Softmax-gated fusion — 각 position마다 uni/bi/tri 채널 중요도 학습.
    기존 단순 concat+Linear보다 dynamic & interpretable.
    """

    def __init__(self, d_model: int):
        super().__init__()
        # concat된 3d를 받아 3-way softmax gate 생성
        self.gate = nn.Linear(d_model * 3, 3)
        self.proj = nn.Linear(d_model, d_model)

    def forward(self, e_uni, e_bi, e_tri):
        # [B, L, 3, d]
        stacked = torch.stack([e_uni, e_bi, e_tri], dim=-2)
        combined = torch.cat([e_uni, e_bi, e_tri], dim=-1)          # [B, L, 3d]
        gates = torch.softmax(self.gate(combined), dim=-1)            # [B, L, 3]
        # weighted sum → [B, L, d]
        fused = (stacked * gates.unsqueeze(-1)).sum(dim=-2)
        return self.proj(fused)


class LinguisticAwareFusion(nn.Module):
    """
    Soft linguistic-prior 기반 n-gram fusion. (논문 Architecture novelty)

    아이디어:
      1. Pre-fusion 임베딩을 mean-pool → auxiliary classifier로 언어권 분포 추정
      2. 언어권별 (uni, bi, tri) prior weights를 학습 (lang_prior 파라미터)
      3. 추정된 lang 분포로 prior를 가중평균 → soft fusion weights
      4. soft weights로 채널 가중합

    Why this is novel:
      · Standard multichannel fusion은 데이터 신호로만 학습 (linguistic prior 0)
      · 슬라브 surnames은 trigram (-ova)이 핵심, 영미 짧은 이름은 unigram도 도움
      · 본 fusion은 "언어권에 따른 n-gram scale 선호"를 명시적으로 모델링

    Auxiliary supervision:
      · forward()는 (fused, aux_lang_logits) 반환
      · trainer가 aux_lang_logits에 추가 lang CE loss를 걸어 학습 가속
      · β·L_aux_lang weight (config의 train.aux_lang_weight)로 조절

    학습 후 lang_prior 행렬을 추출하면 언어권별 fusion preference 분석 가능
    (예: 슬라브 → tri 선호, 영미 → uni 균형) — 정성 분석 가능.
    """

    def __init__(self, d_model: int, n_lang_groups: int, n_channels: int = 3,
                 typology_init: dict = None, lang_idx_map: dict = None):
        """
        Args:
            typology_init: {lang_name: {'uni': p1, 'bi': p2, 'tri': p3}}
                           기존 PE의 sinusoidal init과 같은 역할 (Vaswani-style prior).
                           제공 시 lang_prior matrix를 typology log-probability로 초기화.
            lang_idx_map:  {lang_name: idx} — typology 키와 행 인덱스 매핑
                           (data.get_lang_group_idx(config) 결과)
            둘 중 하나라도 None이면 기존 uniform 초기화 (zeros).
        """
        super().__init__()
        self.n_channels = n_channels
        self.n_lang_groups = n_lang_groups

        # 1. Auxiliary lang classifier (pre-fusion concat → lang dist)
        self.lang_classifier = nn.Sequential(
            nn.Linear(d_model * n_channels, d_model),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(d_model, n_lang_groups),
        )

        # 2. Language-specific fusion priors
        #    typology_init 제공 시 → 언어학적 prior로 초기화 (PE의 sinusoidal init과 동등)
        #    아니면 → uniform (zeros, softmax 후 1/3씩)
        init_matrix = self._build_init_matrix(
            n_lang_groups, n_channels, typology_init, lang_idx_map,
        )
        self.lang_prior = nn.Parameter(init_matrix)
        # 디버그/분석용으로 초기 값 보관 (학습 후 변화 비교 가능)
        self.register_buffer('lang_prior_init', init_matrix.detach().clone())

        # 3. Final projection (다른 fusion과 dim 일치)
        self.proj = nn.Linear(d_model, d_model)

    @staticmethod
    def _build_init_matrix(n_lang_groups, n_channels, typology_init, lang_idx_map):
        """typology dict + idx map → softmax-pre-image 초기화 matrix."""
        if typology_init is None or lang_idx_map is None:
            return torch.zeros(n_lang_groups, n_channels)

        # softmax-pre-image: log(weight)로 변환 (softmax 후 weight 복원)
        init = torch.zeros(n_lang_groups, n_channels)
        keys = ('uni', 'bi', 'tri')
        n_filled = 0
        for lang_name, idx in lang_idx_map.items():
            if lang_name in typology_init and idx < n_lang_groups:
                weights = typology_init[lang_name]
                if isinstance(weights, dict):
                    weights = [weights.get(k, 1.0 / 3) for k in keys]
                init[idx] = torch.tensor(
                    [math.log(max(w, 1e-6)) for w in weights[:n_channels]]
                )
                n_filled += 1
        if n_filled == 0:
            return torch.zeros(n_lang_groups, n_channels)
        return init

    def forward(self, e_uni, e_bi, e_tri, padding_mask=None):
        """
        Returns:
            fused: [B, L, d]
            aux_lang_logits: [B, n_lang_groups] — auxiliary lang prediction
        """
        B, L, d = e_uni.shape

        # 1. Pre-fusion mean pool (padding 제외)
        x_concat = torch.cat([e_uni, e_bi, e_tri], dim=-1)        # [B, L, 3d]
        if padding_mask is not None:
            mask = (~padding_mask).unsqueeze(-1).float()           # [B, L, 1]
            mean_pooled = (x_concat * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        else:
            mean_pooled = x_concat.mean(dim=1)                     # [B, 3d]

        # 2. Lang distribution 추정 (soft)
        aux_lang_logits = self.lang_classifier(mean_pooled)         # [B, n_lang]
        lang_dist = torch.softmax(aux_lang_logits, dim=-1)          # [B, n_lang]

        # 3. Soft fusion prior: 추정 분포로 lang_prior 가중평균
        prior_weights = torch.softmax(self.lang_prior, dim=-1)       # [n_lang, 3]
        soft_prior = lang_dist @ prior_weights                       # [B, 3]

        # 4. Channel-wise weighted fusion ([B, 1, 1] broadcast)
        w_uni = soft_prior[:, 0].view(B, 1, 1)
        w_bi  = soft_prior[:, 1].view(B, 1, 1)
        w_tri = soft_prior[:, 2].view(B, 1, 1)

        fused = w_uni * e_uni + w_bi * e_bi + w_tri * e_tri          # [B, L, d]
        return self.proj(fused), aux_lang_logits


class MultiChannelEncoder(nn.Module):
    """
    Multi-channel (uni + bi + tri) Transformer Encoder. 주력 모델.

    출력 (기본): (nat_logits, lang_logits)
    출력 (return_features=True): (nat_logits, lang_logits, pooled)
      · pooled는 contrastive loss 등에 사용

    Options:
      · fusion_type: 'linear' (기존) | 'gated' | 'linguistic' (Architecture novelty)
      · pool_type:   'attention' (기존) | 'mean' | 'max' | 'cls'
    """

    def __init__(self, n_classes: int, n_lang_groups: int, d_model: int, n_head: int,
                 d_hid: int, n_layers: int, vocab_size: dict, pad_token_id=0,
                 dropout: float = 0.5,
                 fusion_type: str = 'linear',
                 pool_type: str = 'attention',
                 typology_init: dict = None,
                 lang_idx_map: dict = None):
        """
        typology_init / lang_idx_map: linguistic fusion에서만 사용.
          · None이면 lang_prior가 uniform 초기화
          · 제공 시 LinguisticAwareFusion이 typology log-probability로 초기화
        """
        super().__init__()
        self.d_model = d_model
        self.pad_token_id = pad_token_id
        self.fusion_type = fusion_type
        self.pool_type = pool_type
        # 'linguistic' fusion에서 trainer가 fetch할 aux_lang_logits stash
        self._last_aux_lang_logits = None

        # 1. 채널별 독립 임베딩
        self.emb_uni = nn.Embedding(vocab_size['uni'], d_model)
        self.emb_bi = nn.Embedding(vocab_size['bi'], d_model)
        self.emb_tri = nn.Embedding(vocab_size['tri'], d_model)

        # 2. Fusion — linear (기존) | gated | linguistic (신규, Architecture novelty)
        if fusion_type == 'linear':
            self.fusion = nn.Linear(d_model * 3, d_model)
        elif fusion_type == 'gated':
            self.fusion = GatedFusion(d_model)
        elif fusion_type == 'linguistic':
            self.fusion = LinguisticAwareFusion(
                d_model, n_lang_groups,
                typology_init=typology_init,
                lang_idx_map=lang_idx_map,
            )
        else:
            raise ValueError(
                f"fusion_type must be 'linear' | 'gated' | 'linguistic', got {fusion_type}"
            )

        self.pos_encoder = PositionalEncoding(d_model, dropout)

        # 3. Pooling — CLS 사용 시 learnable token 추가
        if pool_type == 'cls':
            self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        else:
            self.cls_token = None

        encoder_layer = nn.TransformerEncoderLayer(
            d_model, n_head, d_hid, dropout, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, n_layers)

        # Attention pooling은 module 필요, 나머지는 operation
        if pool_type == 'attention':
            self.pooling = AttentionPooling(d_model)
        else:
            self.pooling = None   # functional pooling (mean/max/cls)

        # Dual head
        self.head_nat = nn.Linear(d_model, n_classes)
        self.head_lang = nn.Linear(d_model, n_lang_groups)

    def _fuse(self, e_uni, e_bi, e_tri, padding_mask=None):
        """
        Fusion 로직 (linear / gated / linguistic 통합).
        linguistic은 (fused, aux_lang_logits) 반환, 나머지는 (fused, None).
        """
        if self.fusion_type == 'linear':
            combined = torch.cat([e_uni, e_bi, e_tri], dim=-1)
            return self.fusion(combined), None
        if self.fusion_type == 'gated':
            return self.fusion(e_uni, e_bi, e_tri), None
        # linguistic — aux_lang_logits 함께 반환
        return self.fusion(e_uni, e_bi, e_tri, padding_mask)

    def _pool(self, x, padding_mask):
        """Pooling 로직 (attention/mean/max/cls 통합)."""
        if self.pool_type == 'attention':
            return self.pooling(x, mask=padding_mask)
        if self.pool_type == 'mean':
            # padding 위치 제외한 평균
            valid = (~padding_mask).unsqueeze(-1).float()
            return (x * valid).sum(dim=1) / (valid.sum(dim=1).clamp(min=1.0))
        if self.pool_type == 'max':
            x_masked = x.masked_fill(padding_mask.unsqueeze(-1), -1e4)
            return x_masked.max(dim=1).values
        if self.pool_type == 'cls':
            return x[:, 0, :]   # CLS는 0번 position
        raise ValueError(f"unknown pool_type: {self.pool_type}")

    def forward(self, x_uni, x_bi, x_tri, return_features: bool = False):
        # Padding mask는 uni 기준
        padding_mask = (x_uni == self.pad_token_id)

        e_uni = self.emb_uni(x_uni)
        e_bi = self.emb_bi(x_bi)
        e_tri = self.emb_tri(x_tri)

        fused, aux_lang_logits = self._fuse(e_uni, e_bi, e_tri, padding_mask)
        # linguistic fusion일 때만 aux_lang_logits 존재 — trainer가 fetch
        self._last_aux_lang_logits = aux_lang_logits

        x = fused * math.sqrt(self.d_model)

        # CLS token prepend (pool_type='cls'일 때)
        if self.cls_token is not None:
            B = x.size(0)
            cls = self.cls_token.expand(B, -1, -1)
            x = torch.cat([cls, x], dim=1)
            # padding_mask도 CLS 위치(False) 추가
            cls_mask = torch.zeros(B, 1, dtype=torch.bool, device=padding_mask.device)
            padding_mask = torch.cat([cls_mask, padding_mask], dim=1)

        x = self.pos_encoder(x)
        x = self.transformer_encoder(x, src_key_padding_mask=padding_mask)
        pooled = self._pool(x, padding_mask)

        nat_logits = self.head_nat(pooled)
        lang_logits = self.head_lang(pooled)

        if return_features:
            return nat_logits, lang_logits, pooled
        return nat_logits, lang_logits

    def encode(self, x_uni, x_bi, x_tri):
        """
        분류 head 이전의 pooled representation 반환.
        Phase 4 분석(embedding 거리 vs CNL 상관관계)에서 사용.
        """
        with torch.no_grad():
            _, _, pooled = self.forward(x_uni, x_bi, x_tri, return_features=True)
        return pooled


class AttentionPooling(nn.Module):
    """시퀀스 출력을 attention weighted sum으로 집약."""

    def __init__(self, d_model):
        super().__init__()
        self.attn = nn.Linear(d_model, 1)

    def forward(self, x, mask=None):
        # x: [batch, seq_len, d_model]
        attn_weights = self.attn(x)  # [batch, seq_len, 1]
        if mask is not None:
            # 패딩 위치의 attention score를 아주 작게 → softmax 이후 0
            attn_weights = attn_weights.masked_fill(mask.unsqueeze(-1), -1e4)
        attn_weights = torch.softmax(attn_weights, dim=1)
        return torch.sum(attn_weights * x, dim=1)  # [batch, d_model]


class PositionalEncoding(nn.Module):
    """
    Sinusoidal Positional Encoding (Vaswani et al., 2017).
    batch_first=True: 입력/PE 모두 [batch, seq_len, d_model] 형태.
    """

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)  # [max_len, 1]
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )  # [d_model / 2]

        pe = torch.zeros(max_len, d_model)             # [max_len, d_model]
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)                           # [1, max_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: [batch, seq_len, d_model]
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


# ─────────────────────────────────────────────────────────────────────
# Conv Stem — character-level local pattern extractor before attention.
# CNN의 inductive bias (translation invariance + locality)를 명시적 주입.
# Embedding → ConvStem → PositionalEncoding → Conformer 순서.
# ─────────────────────────────────────────────────────────────────────

class ConvStemBlock(nn.Module):
    """
    Single depthwise-separable conv block — pre-norm + residual.

    구성: LN → depthwise Conv1d → pointwise Conv1d → GELU → Dropout → +residual
    Sequence-length 보존 (padding = kernel_size // 2).
    """

    def __init__(self, d_model: int, kernel_size: int = 5, dropout: float = 0.1):
        super().__init__()
        if kernel_size % 2 == 0:
            kernel_size += 1   # 홀수 강제 (대칭 패딩)
        self.norm = nn.LayerNorm(d_model)
        # depthwise — 채널별 독립 local pattern 추출
        self.depthwise = nn.Conv1d(
            d_model, d_model, kernel_size,
            padding=kernel_size // 2, groups=d_model,
        )
        # pointwise — 채널 간 mixing
        self.pointwise = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: [B, L, d]  (batch_first)
        residual = x
        h = self.norm(x)
        h = h.transpose(1, 2)              # [B, d, L]
        h = self.depthwise(h)
        h = self.pointwise(h)
        h = h.transpose(1, 2)              # [B, L, d]
        h = self.act(h)
        h = self.dropout(h)
        return residual + h


class ConvStem(nn.Module):
    """
    Conv stem — n_layers개의 depthwise-separable conv 블록 stack.

    Why: character embedding을 "n-gram detector" feature로 미리 변환해서
    self-attention이 풍부한 local feature 위에서 동작하게 함.
    ConvNeXt/ConvViT의 motivation과 동일.

    Embedding 직후, PositionalEncoding 전에 삽입.
    Sequence length 보존 → MCC의 cross-attn / pooling과 호환.
    """

    def __init__(self, d_model: int, kernel_size: int = 5, n_layers: int = 2,
                 dropout: float = 0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            ConvStemBlock(d_model, kernel_size, dropout)
            for _ in range(n_layers)
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class MultiScaleConvStemBlock(nn.Module):
    """
    Inception-style multi-scale conv stem block.

    구성: LN → [k=3, k=5, k=7 병렬 depthwise convs] → concat → 1x1 proj → GELU → +residual

    Why: CNN baseline은 보통 multi-kernel을 동시에 사용 (Inception, ConvNeXt 등).
    여러 receptive field를 한 layer에서 동시에 추출 → 이 task처럼 다양한
    n-gram scale 신호가 섞여있는 데이터에 적합.
    """

    def __init__(self, d_model: int, kernel_sizes=(3, 5, 7), dropout: float = 0.1):
        super().__init__()
        # 모든 kernel을 홀수로 (대칭 패딩)
        kernel_sizes = tuple(k if k % 2 == 1 else k + 1 for k in kernel_sizes)
        self.norm = nn.LayerNorm(d_model)
        # 여러 kernel size로 병렬 depthwise conv
        self.convs = nn.ModuleList([
            nn.Conv1d(d_model, d_model, k, padding=k // 2, groups=d_model)
            for k in kernel_sizes
        ])
        # concat된 multi-scale feature를 d_model로 fuse (pointwise)
        self.proj = nn.Conv1d(d_model * len(kernel_sizes), d_model, kernel_size=1)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: [B, L, d]
        residual = x
        h = self.norm(x).transpose(1, 2)              # [B, d, L]
        # Parallel multi-scale convs
        outs = [conv(h) for conv in self.convs]
        h = torch.cat(outs, dim=1)                     # [B, d * n_kernels, L]
        h = self.proj(h)                                # [B, d, L]
        h = h.transpose(1, 2)                           # [B, L, d]
        h = self.act(h)
        h = self.dropout(h)
        return residual + h


class MultiScaleConvStem(nn.Module):
    """Multi-scale conv stem — n_layers개의 Inception-style block stack."""

    def __init__(self, d_model: int, kernel_sizes=(3, 5, 7),
                 n_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            MultiScaleConvStemBlock(d_model, kernel_sizes, dropout)
            for _ in range(n_layers)
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


# ─────────────────────────────────────────────────────────────────────
# Multi-Channel Conformer (MCC) — Architecture novelty for short paper.
# Conformer (Gulati et al., Interspeech 2020)을 multichannel character
# NLP에 적용. 각 n-gram 채널이 specialized Conformer block을 거치고
# cross-channel attention으로 통합.
# ─────────────────────────────────────────────────────────────────────

class MacaronFFN(nn.Module):
    """½-scale Feed-Forward Network — Conformer의 macaron-style FFN."""

    def __init__(self, d_model: int, d_hid: int, dropout: float = 0.1):
        super().__init__()
        self.layer_norm = nn.LayerNorm(d_model)
        self.linear1 = nn.Linear(d_model, d_hid)
        self.swish = nn.SiLU()
        self.dropout1 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_hid, d_model)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        # ½-scale residual: x + 0.5 * FFN(x)
        residual = x
        h = self.layer_norm(x)
        h = self.linear1(h)
        h = self.swish(h)
        h = self.dropout1(h)
        h = self.linear2(h)
        h = self.dropout2(h)
        return residual + 0.5 * h


class ConformerConvModule(nn.Module):
    """
    Conformer convolution module (Gulati et al., 2020).

    구성: LayerNorm → Pointwise Conv (expansion 2) → GLU → Depthwise Conv
          → BatchNorm → Swish → Pointwise Conv → Dropout → residual
    """

    def __init__(self, d_model: int, kernel_size: int = 15,
                 dropout: float = 0.1):
        super().__init__()
        # kernel_size를 홀수로 정규화 (depthwise conv padding 대칭 위해)
        if kernel_size % 2 == 0:
            kernel_size += 1
        self.layer_norm = nn.LayerNorm(d_model)
        self.pointwise_conv1 = nn.Conv1d(d_model, 2 * d_model, kernel_size=1)
        self.glu = nn.GLU(dim=1)
        self.depthwise_conv = nn.Conv1d(
            d_model, d_model, kernel_size,
            padding=kernel_size // 2,
            groups=d_model,
        )
        self.batch_norm = nn.BatchNorm1d(d_model)
        self.swish = nn.SiLU()
        self.pointwise_conv2 = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: [B, L, d]
        residual = x
        h = self.layer_norm(x).transpose(1, 2)         # [B, d, L]
        h = self.pointwise_conv1(h)                    # [B, 2d, L]
        h = self.glu(h)                                 # [B, d, L]
        h = self.depthwise_conv(h)                      # [B, d, L]
        h = self.batch_norm(h)
        h = self.swish(h)
        h = self.pointwise_conv2(h)                     # [B, d, L]
        h = h.transpose(1, 2)                           # [B, L, d]
        h = self.dropout(h)
        return residual + h


class ConformerBlock(nn.Module):
    """
    Conformer block: ½FFN → MHSA → Conv → ½FFN → LayerNorm.
    Phase 4 호환을 위해 last MHSA attention weights 보관 (`_last_attn`).
    """

    def __init__(self, d_model: int, n_head: int, d_hid: int,
                 conv_kernel: int = 15, dropout: float = 0.1):
        super().__init__()
        self.ffn1 = MacaronFFN(d_model, d_hid, dropout)
        self.mhsa_norm = nn.LayerNorm(d_model)
        self.mhsa = nn.MultiheadAttention(
            d_model, n_head, dropout=dropout, batch_first=True
        )
        self.mhsa_dropout = nn.Dropout(dropout)
        self.conv = ConformerConvModule(d_model, conv_kernel, dropout)
        self.ffn2 = MacaronFFN(d_model, d_hid, dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        # Phase 4 분석용 — 마지막 forward의 attention weights stash
        self._last_attn = None

    def forward(self, x, src_key_padding_mask=None):
        # ½ FFN
        x = self.ffn1(x)
        # MHSA
        residual = x
        h = self.mhsa_norm(x)
        attn_out, attn_w = self.mhsa(
            h, h, h,
            key_padding_mask=src_key_padding_mask,
            need_weights=True,
            average_attn_weights=True,
        )
        self._last_attn = attn_w
        x = residual + self.mhsa_dropout(attn_out)
        # Conv
        x = self.conv(x)
        # ½ FFN
        x = self.ffn2(x)
        # Final LayerNorm
        return self.layer_norm(x)


class MultiChannelConformer(nn.Module):
    """
    Multi-Channel Conformer (MCC) — 본 페이퍼의 architecture novelty.

    설계 결정 3가지:
      1. **채널별 Conformer stack** (uni/bi/tri 각자 specialized)
      2. **채널별 다른 conv kernel size** (uni→2, bi→3, tri→5)
         각 n-gram scale에 맞는 receptive field
      3. **Cross-channel attention** — 채널 간 morphological dependency 학습

    Plain Conformer (single-stream) 대비 multichannel character NLP에
    task-specific하게 최적화. Vanilla Transformer 대비 conv module로
    local pattern modeling 보강 (CNN 강점 차용). Self-attention 보존
    → Phase 4 trigram 분석 그대로 가능.
    """

    def __init__(self, n_classes: int, n_lang_groups: int, d_model: int,
                 n_head: int, d_hid: int, n_layers: int, vocab_size: dict,
                 pad_token_id: int = 0, dropout: float = 0.5,
                 conv_kernels=(2, 3, 5),    # uni / bi / tri
                 pool_type: str = 'attention',
                 cross_attn_per_layer: bool = False,
                 conv_stem: bool = False,
                 conv_stem_kernel: int = 5,
                 conv_stem_layers: int = 2,
                 conv_stem_type: str = 'standard',   # 'standard' | 'multiscale'
                 conv_stem_kernels=(3, 5, 7),         # multiscale 전용
                 channel_ensemble_head: bool = False,
                 channel_fusion_type: str = 'uniform',  # 'uniform' | 'gate'
                 **kwargs):
        super().__init__()
        self.d_model = d_model
        self.n_layers = n_layers
        self.pad_token_id = pad_token_id
        self.pool_type = pool_type
        self.cross_attn_per_layer = bool(cross_attn_per_layer)
        self.use_conv_stem = bool(conv_stem)
        self.channel_ensemble_head = bool(channel_ensemble_head)
        self.channel_fusion_type = channel_fusion_type

        # 1. 채널별 임베딩
        self.emb_uni = nn.Embedding(vocab_size['uni'], d_model)
        self.emb_bi  = nn.Embedding(vocab_size['bi'],  d_model)
        self.emb_tri = nn.Embedding(vocab_size['tri'], d_model)

        # 1.5. Conv stem (선택) — 채널별 독립 local pattern extractor
        # · standard: 단일 kernel depthwise-separable
        # · multiscale: [3,5,7] 병렬 (Inception-style)
        if self.use_conv_stem:
            stem_cls = (MultiScaleConvStem if conv_stem_type == 'multiscale'
                        else ConvStem)
            stem_kwargs = dict(d_model=d_model, n_layers=conv_stem_layers, dropout=dropout)
            if conv_stem_type == 'multiscale':
                stem_kwargs['kernel_sizes'] = tuple(conv_stem_kernels)
            else:
                stem_kwargs['kernel_size'] = conv_stem_kernel
            self.stem_uni = stem_cls(**stem_kwargs)
            self.stem_bi  = stem_cls(**stem_kwargs)
            self.stem_tri = stem_cls(**stem_kwargs)
        else:
            self.stem_uni = self.stem_bi = self.stem_tri = None

        # 2. Positional encoding (shared)
        self.pos_encoder = PositionalEncoding(d_model, dropout)

        # 3. 채널별 Conformer stack with channel-specific conv kernels
        kuni, kbi, ktri = conv_kernels
        self.conformer_uni = nn.ModuleList([
            ConformerBlock(d_model, n_head, d_hid, kuni, dropout)
            for _ in range(n_layers)
        ])
        self.conformer_bi = nn.ModuleList([
            ConformerBlock(d_model, n_head, d_hid, kbi, dropout)
            for _ in range(n_layers)
        ])
        self.conformer_tri = nn.ModuleList([
            ConformerBlock(d_model, n_head, d_hid, ktri, dropout)
            for _ in range(n_layers)
        ])

        # 4. Cross-channel attention
        # · 기본: 마지막에 1번 (concat → self-attn → uni-aligned 추출)
        # · cross_attn_per_layer=True: 매 layer마다 mix → channel-fragmentation 완화
        #   (가설: ρ 회복 + classification 시너지)
        if self.cross_attn_per_layer:
            self.cross_attns = nn.ModuleList([
                nn.MultiheadAttention(d_model, n_head, dropout=dropout, batch_first=True)
                for _ in range(n_layers)
            ])
            self.cross_norms = nn.ModuleList([
                nn.LayerNorm(d_model) for _ in range(n_layers)
            ])
            self.cross_attn = None
            self.cross_norm = None
        else:
            self.cross_attn = nn.MultiheadAttention(
                d_model, n_head, dropout=dropout, batch_first=True
            )
            self.cross_norm = nn.LayerNorm(d_model)
            self.cross_attns = None
            self.cross_norms = None

        # 5. Pooling
        if pool_type == 'attention':
            self.pooling = AttentionPooling(d_model)
        else:
            self.pooling = None

        # 6. Dual head
        # · 기본: 단일 dual head (cross-attn 후 fused representation에서)
        # · channel_ensemble_head=True: 채널별 독립 dual head + 평균 fusion
        #   → cross-attn 우회. 각 채널이 완전한 분류기 (bagging-like ensemble)
        if self.channel_ensemble_head:
            # 채널별 독립 pooling (각자의 attention 패턴 학습)
            if pool_type == 'attention':
                self.pool_uni = AttentionPooling(d_model)
                self.pool_bi  = AttentionPooling(d_model)
                self.pool_tri = AttentionPooling(d_model)
            else:
                self.pool_uni = self.pool_bi = self.pool_tri = None
            # 채널별 독립 dual head
            self.head_nat_uni  = nn.Linear(d_model, n_classes)
            self.head_nat_bi   = nn.Linear(d_model, n_classes)
            self.head_nat_tri  = nn.Linear(d_model, n_classes)
            self.head_lang_uni = nn.Linear(d_model, n_lang_groups)
            self.head_lang_bi  = nn.Linear(d_model, n_lang_groups)
            self.head_lang_tri = nn.Linear(d_model, n_lang_groups)
            # placeholder (unused but model state 유지)
            self.head_nat  = nn.Linear(d_model, n_classes)
            self.head_lang = nn.Linear(d_model, n_lang_groups)
            # Input-dependent gate (MoE-style fusion)
            # · uniform: 1/3씩 평균 (기존)
            # · gate:    [p_uni;p_bi;p_tri] → MLP → softmax → 샘플별 채널 가중치
            if self.channel_fusion_type == 'gate':
                self.channel_gate = nn.Sequential(
                    nn.Linear(d_model * 3, d_model),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(d_model, 3),
                )
            else:
                self.channel_gate = None
        else:
            self.head_nat  = nn.Linear(d_model, n_classes)
            self.head_lang = nn.Linear(d_model, n_lang_groups)

    def _encode_channel(self, blocks, x, padding_mask):
        for block in blocks:
            x = block(x, src_key_padding_mask=padding_mask)
        return x

    def _pool(self, x, padding_mask):
        if self.pool_type == 'attention':
            return self.pooling(x, mask=padding_mask)
        # mean fallback
        valid = (~padding_mask).unsqueeze(-1).float()
        return (x * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1.0)

    def _pool_channel(self, x, padding_mask, channel: str):
        """채널별 독립 pooling (channel_ensemble_head 모드 전용)."""
        if self.pool_type == 'attention':
            pool_module = {'uni': self.pool_uni, 'bi': self.pool_bi, 'tri': self.pool_tri}[channel]
            return pool_module(x, mask=padding_mask)
        # mean fallback
        valid = (~padding_mask).unsqueeze(-1).float()
        return (x * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1.0)

    def forward(self, x_uni, x_bi, x_tri, return_features: bool = False):
        # Padding mask는 uni 기준 (모든 채널 동일 길이 가정)
        padding_mask = (x_uni == self.pad_token_id)

        # 1. 채널별 임베딩 + scale
        scale = math.sqrt(self.d_model)
        h_uni = self.emb_uni(x_uni) * scale
        h_bi  = self.emb_bi(x_bi)   * scale
        h_tri = self.emb_tri(x_tri) * scale

        # 1.5. Conv stem (선택) — 채널별 local pattern 추출 (attention 전에 적용)
        if self.use_conv_stem:
            h_uni = self.stem_uni(h_uni)
            h_bi  = self.stem_bi(h_bi)
            h_tri = self.stem_tri(h_tri)

        # 2. Positional encoding (shared)
        h_uni = self.pos_encoder(h_uni)
        h_bi  = self.pos_encoder(h_bi)
        h_tri = self.pos_encoder(h_tri)

        L = h_uni.size(1)
        cat_mask = torch.cat([padding_mask] * 3, dim=1)              # [B, 3L]

        # 3. 채널별 Conformer stacks (with optional per-layer cross-attn)
        if self.cross_attn_per_layer:
            for i in range(self.n_layers):
                h_uni = self.conformer_uni[i](h_uni, src_key_padding_mask=padding_mask)
                h_bi  = self.conformer_bi[i](h_bi,   src_key_padding_mask=padding_mask)
                h_tri = self.conformer_tri[i](h_tri, src_key_padding_mask=padding_mask)

                cat = torch.cat([h_uni, h_bi, h_tri], dim=1)
                cross_out, _ = self.cross_attns[i](
                    cat, cat, cat,
                    key_padding_mask=cat_mask,
                    need_weights=False,
                )
                cat = self.cross_norms[i](cat + cross_out)
                h_uni = cat[:, :L, :]
                h_bi  = cat[:, L:2 * L, :]
                h_tri = cat[:, 2 * L:, :]
        else:
            h_uni = self._encode_channel(self.conformer_uni, h_uni, padding_mask)
            h_bi  = self._encode_channel(self.conformer_bi,  h_bi,  padding_mask)
            h_tri = self._encode_channel(self.conformer_tri, h_tri, padding_mask)

        # 4. Output: 3-way 분기
        if self.channel_ensemble_head:
            # 각 채널 → 독립 pool + 독립 head → logits 결합 (uniform 평균 또는 gate)
            p_uni = self._pool_channel(h_uni, padding_mask, 'uni')
            p_bi  = self._pool_channel(h_bi,  padding_mask, 'bi')
            p_tri = self._pool_channel(h_tri, padding_mask, 'tri')

            nat_uni  = self.head_nat_uni(p_uni)
            nat_bi   = self.head_nat_bi(p_bi)
            nat_tri  = self.head_nat_tri(p_tri)
            lang_uni = self.head_lang_uni(p_uni)
            lang_bi  = self.head_lang_bi(p_bi)
            lang_tri = self.head_lang_tri(p_tri)

            if self.channel_fusion_type == 'gate':
                # 샘플마다 다른 채널 가중치 (MoE-style)
                gate_input = torch.cat([p_uni, p_bi, p_tri], dim=-1)     # [B, 3d]
                gate = F.softmax(self.channel_gate(gate_input), dim=-1)  # [B, 3]
                g_uni = gate[:, 0:1]
                g_bi  = gate[:, 1:2]
                g_tri = gate[:, 2:3]
                nat_logits  = g_uni * nat_uni + g_bi * nat_bi + g_tri * nat_tri
                lang_logits = g_uni * lang_uni + g_bi * lang_bi + g_tri * lang_tri
                # 분석/regularization 용 (entropy reg 등에서 참조 가능)
                self._last_gate = gate
            else:
                # uniform 평균 (기존 bagging-like)
                nat_logits  = (nat_uni + nat_bi + nat_tri) / 3.0
                lang_logits = (lang_uni + lang_bi + lang_tri) / 3.0
            pooled = (p_uni + p_bi + p_tri) / 3.0
        elif self.cross_attn_per_layer:
            # 매 layer cross-attn으로 이미 채널 mix됨 → h_uni 그대로 사용
            pooled = self._pool(h_uni, padding_mask)
            nat_logits  = self.head_nat(pooled)
            lang_logits = self.head_lang(pooled)
        else:
            # 기본: 마지막 cross-attn 1회 → fused → single head
            combined = torch.cat([h_uni, h_bi, h_tri], dim=1)         # [B, 3L, d]
            cross_out, _ = self.cross_attn(
                combined, combined, combined,
                key_padding_mask=cat_mask,
                need_weights=False,
            )
            h_fused = self.cross_norm(h_uni + cross_out[:, :L, :])
            pooled = self._pool(h_fused, padding_mask)
            nat_logits  = self.head_nat(pooled)
            lang_logits = self.head_lang(pooled)

        if return_features:
            return nat_logits, lang_logits, pooled
        return nat_logits, lang_logits

    def encode(self, x_uni, x_bi, x_tri):
        """Phase 4 분석용 — fused pooled representation 반환."""
        with torch.no_grad():
            _, _, pooled = self.forward(x_uni, x_bi, x_tri, return_features=True)
        return pooled
