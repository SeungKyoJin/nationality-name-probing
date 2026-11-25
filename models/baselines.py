"""
비교 모델 구현 (Baseline Models).

모든 모델은 MultiChannelEncoder와 동일한 입력(uni, bi, tri)을 받고
동일한 출력(nat_logits, lang_logits)을 반환한다.
이렇게 하면 동일 데이터 파이프라인과 Trainer를 그대로 재사용할 수 있다.

입력 처리 전략:
  3개 채널 임베딩 → concat → fusion linear (d_model*3 → d_model) → 각 모델의 core
  이는 MultiChannelEncoder의 fusion 방식과 동일하여 공정 비교가 가능하다.
"""

import math

import torch
import torch.nn as nn


class MultiChannelCNN(nn.Module):
    """
    Multi-channel Character CNN (Kim, 2014 스타일).

    각 채널(uni/bi/tri) 임베딩 → fusion → 다중 커널 Conv1d → MaxPool → dual head.
    """

    def __init__(self, n_classes, n_lang_groups, d_model, d_hid,
                 n_head, n_layers,  # Trainer 호환용 (사용하지 않음)
                 vocab_size, pad_token_id=0, dropout=0.3):
        super().__init__()
        self.pad_token_id = pad_token_id

        # 채널별 임베딩
        self.emb_uni = nn.Embedding(vocab_size['uni'], d_model)
        self.emb_bi = nn.Embedding(vocab_size['bi'], d_model)
        self.emb_tri = nn.Embedding(vocab_size['tri'], d_model)
        self.fusion = nn.Linear(d_model * 3, d_model)

        # 다중 커널 Conv1d
        filter_sizes = [2, 3, 4, 5]
        num_filters = d_model // 2  # 각 커널당 필터 수
        self.convs = nn.ModuleList([
            nn.Conv1d(d_model, num_filters, fs, padding=fs // 2)
            for fs in filter_sizes
        ])
        self.dropout = nn.Dropout(dropout)

        pooled_dim = num_filters * len(filter_sizes)
        self.fc = nn.Linear(pooled_dim, d_model)

        # Dual head
        self.head_nat = nn.Linear(d_model, n_classes)
        self.head_lang = nn.Linear(d_model, n_lang_groups)

    def encode(self, x_uni, x_bi, x_tri):
        """분류 head 직전의 pooled representation 반환.

        DICL 상관관계 분석(analysis/dicl_correlation.py)에서 사용.
        forward()와 동일한 흐름을 공유하되 head 호출만 생략한다.
        """
        e_uni = self.emb_uni(x_uni)
        e_bi = self.emb_bi(x_bi)
        e_tri = self.emb_tri(x_tri)

        x = self.fusion(torch.cat([e_uni, e_bi, e_tri], dim=-1))
        x = x.transpose(1, 2)

        conv_outs = []
        for conv in self.convs:
            c = torch.relu(conv(x))
            c = c.max(dim=2).values
            conv_outs.append(c)

        pooled = torch.cat(conv_outs, dim=1)
        pooled = self.dropout(pooled)
        return torch.relu(self.fc(pooled))    # [B, d_model]

    def forward(self, x_uni, x_bi, x_tri):
        h = self.encode(x_uni, x_bi, x_tri)
        return self.head_nat(h), self.head_lang(h)


class MultiChannelBiLSTM(nn.Module):
    """
    Multi-channel Character BiLSTM.

    각 채널 임베딩 → fusion → BiLSTM → 마지막 hidden state → dual head.
    """

    def __init__(self, n_classes, n_lang_groups, d_model, d_hid,
                 n_head, n_layers,  # Trainer 호환용
                 vocab_size, pad_token_id=0, dropout=0.3):
        super().__init__()
        self.pad_token_id = pad_token_id

        # 채널별 임베딩
        self.emb_uni = nn.Embedding(vocab_size['uni'], d_model)
        self.emb_bi = nn.Embedding(vocab_size['bi'], d_model)
        self.emb_tri = nn.Embedding(vocab_size['tri'], d_model)
        self.fusion = nn.Linear(d_model * 3, d_model)

        # BiLSTM
        self.lstm = nn.LSTM(
            input_size=d_model,
            hidden_size=d_model // 2,  # 양방향이므로 최종 d_model
            num_layers=n_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)

        # Dual head
        self.head_nat = nn.Linear(d_model, n_classes)
        self.head_lang = nn.Linear(d_model, n_lang_groups)

    def encode(self, x_uni, x_bi, x_tri):
        """분류 head 직전의 hidden state(BiLSTM 양방향 concat) 반환."""
        e_uni = self.emb_uni(x_uni)
        e_bi = self.emb_bi(x_bi)
        e_tri = self.emb_tri(x_tri)

        x = self.fusion(torch.cat([e_uni, e_bi, e_tri], dim=-1))

        lengths = (x_uni != self.pad_token_id).sum(dim=1).cpu().clamp(min=1)
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths, batch_first=True, enforce_sorted=False
        )
        _, (h_n, _) = self.lstm(packed)

        h_forward = h_n[-2]
        h_backward = h_n[-1]
        h = torch.cat([h_forward, h_backward], dim=1)
        return self.dropout(h)                # [B, d_model]

    def forward(self, x_uni, x_bi, x_tri):
        h = self.encode(x_uni, x_bi, x_tri)
        return self.head_nat(h), self.head_lang(h)


class MultiChannelFastText(nn.Module):
    """
    Multi-channel FastText-style classifier (Joulin et al., 2017).

    핵심: 임베딩의 평균(mean pooling)만으로 분류.
    단순하지만 subword 정보를 활용하여 강력한 baseline.
    """

    def __init__(self, n_classes, n_lang_groups, d_model, d_hid,
                 n_head, n_layers,  # Trainer 호환용
                 vocab_size, pad_token_id=0, dropout=0.3):
        super().__init__()
        self.pad_token_id = pad_token_id

        # 채널별 임베딩
        self.emb_uni = nn.Embedding(vocab_size['uni'], d_model)
        self.emb_bi = nn.Embedding(vocab_size['bi'], d_model)
        self.emb_tri = nn.Embedding(vocab_size['tri'], d_model)
        self.fusion = nn.Linear(d_model * 3, d_model)

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(d_model, d_model)

        # Dual head
        self.head_nat = nn.Linear(d_model, n_classes)
        self.head_lang = nn.Linear(d_model, n_lang_groups)

    def encode(self, x_uni, x_bi, x_tri):
        """분류 head 직전의 mean-pooled representation 반환."""
        e_uni = self.emb_uni(x_uni)
        e_bi = self.emb_bi(x_bi)
        e_tri = self.emb_tri(x_tri)

        x = self.fusion(torch.cat([e_uni, e_bi, e_tri], dim=-1))

        mask = (x_uni != self.pad_token_id).unsqueeze(-1).float()
        x = (x * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)

        return self.dropout(torch.relu(self.fc(x)))   # [B, d_model]

    def forward(self, x_uni, x_bi, x_tri):
        h = self.encode(x_uni, x_bi, x_tri)
        return self.head_nat(h), self.head_lang(h)
