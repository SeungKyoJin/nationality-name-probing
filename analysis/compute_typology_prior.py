"""
Empirical typology prior 자동 계산기.

각 LanguageGroup g에 대해 (uni, bi, tri) channel prior를 도출:
  1. 각 n ∈ {1, 2, 3}별 top-k discriminative n-gram 점수 계산
  2. 점수를 정규화 → (uni, bi, tri) prior

Discriminativeness 정의:
    disc(g, n) = mean over top-k frequent n-grams of
                 [freq(ng | g) / (freq(ng | ¬g) + ε)]

직관:
  · 슬라브 → tri-discriminative 큼 (`-ova` 같은 어말 마커 흔함)
  · 영미  → uni-discriminative 큼 (짧은 이름의 단일 char도 의미 가짐)

페이퍼 framing:
  "We derive empirical typological priors via top-k discriminative n-gram
   scoring on the training set. This procedure produces priors aligned
   with linguistic intuition (e.g., Slavic groups receive trigram-heavy
   priors matching morphological markers like '-ova')."

Usage:
    python -m analysis.compute_typology_prior \
        --config configs/base_config.yaml \
        --output data/linguistics/typology_prior.json
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from typing import Dict

import pandas as pd
import yaml

from utils.util import replace_disallowed_chars


# 데이터 로딩 — data 패키지 의존
def _load_train_split(config) -> pd.DataFrame:
    """학습 split만 추출 (test 누수 방지)."""
    from data.data_loader import get_loaded_df, robust_split

    df = get_loaded_df(config).copy()
    df['Name'] = df['Name'].apply(replace_disallowed_chars)
    df = df[df['Name'].str.len() > 0].reset_index(drop=True)

    train_idx, _, _ = robust_split(df)
    return df.iloc[train_idx].reset_index(drop=True)


def compute_typology_prior(
    df: pd.DataFrame,
    lang_col: str = 'LanguageGroup',
    name_col: str = 'Name',
    top_k: int = 30,
    min_samples: int = 5,
    eps: float = 1e-6,
) -> Dict[str, Dict[str, float]]:
    """
    각 언어권별 (uni, bi, tri) typology prior 계산.

    Args:
        df: train DataFrame ([Name, LanguageGroup] 포함)
        top_k: 빈도 상위 몇 개 n-gram의 discriminative score 평균낼지
        min_samples: 이 수 미만 클래스는 uniform fallback
        eps: 분모 0 방지

    Returns:
        {lang_group: {'uni': p1, 'bi': p2, 'tri': p3, 'n_samples': N}}
        → fusion 가중치로 직접 사용
    """
    typology = {}

    for lang in sorted(df[lang_col].unique()):
        lang_names = df[df[lang_col] == lang][name_col].tolist()
        other_names = df[df[lang_col] != lang][name_col].tolist()

        if len(lang_names) < min_samples:
            typology[lang] = {
                'uni': 1.0 / 3, 'bi': 1.0 / 3, 'tri': 1.0 / 3,
                'n_samples': len(lang_names),
                'note': f'fallback uniform (n<{min_samples})',
            }
            continue

        scale_scores: Dict[int, float] = {}

        for n in (1, 2, 3):
            # 1) 해당 lang에서의 n-gram 빈도
            lang_ng: Counter = Counter()
            for name in lang_names:
                for i in range(len(name) - n + 1):
                    lang_ng[name[i:i + n]] += 1
            total_lang = sum(lang_ng.values()) or 1

            # 2) 다른 모든 lang에서의 n-gram 빈도
            other_ng: Counter = Counter()
            for name in other_names:
                for i in range(len(name) - n + 1):
                    other_ng[name[i:i + n]] += 1
            total_other = sum(other_ng.values()) or 1

            # 3) Top-k discriminative score (lang_freq / other_freq 평균)
            disc_scores = []
            for ng, cnt in lang_ng.most_common(top_k):
                lang_freq = cnt / total_lang
                other_freq = other_ng.get(ng, 0) / total_other
                disc = lang_freq / (other_freq + eps)
                disc_scores.append(disc)

            scale_scores[n] = sum(disc_scores) / max(len(disc_scores), 1)

        # 4) 정규화 → (uni, bi, tri) prior
        total = sum(scale_scores.values())
        if total > 0:
            typology[lang] = {
                'uni': scale_scores[1] / total,
                'bi':  scale_scores[2] / total,
                'tri': scale_scores[3] / total,
                'n_samples': len(lang_names),
            }
        else:
            typology[lang] = {
                'uni': 1.0 / 3, 'bi': 1.0 / 3, 'tri': 1.0 / 3,
                'n_samples': len(lang_names),
                'note': 'fallback (zero discriminativeness)',
            }

    return typology


def print_typology_table(typology: Dict[str, Dict[str, float]]) -> None:
    """언어권별 prior 표 콘솔 출력 — tri 우세 순으로 정렬."""
    print("\n" + "=" * 70)
    print(" Empirical Typology Prior (sorted by tri-preference)")
    print("=" * 70)
    print(f"{'Language Group':<25s} {'uni':>7s} {'bi':>7s} {'tri':>7s} {'n':>7s}")
    print("-" * 70)

    sorted_typology = sorted(
        typology.items(),
        key=lambda x: x[1].get('tri', 0),
        reverse=True,
    )
    for lang, prior in sorted_typology:
        n = prior.get('n_samples', '?')
        n_str = f"{n:>7d}" if isinstance(n, int) else f"{n:>7s}"
        if 'note' in prior and 'fallback' in prior.get('note', ''):
            print(f"{lang:<25s} {'-':>7s} {'-':>7s} {'-':>7s} {n_str}  ({prior['note']})")
        else:
            print(f"{lang:<25s} {prior['uni']:>7.3f} {prior['bi']:>7.3f} "
                  f"{prior['tri']:>7.3f} {n_str}")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument('--config', default='configs/base_config.yaml')
    parser.add_argument('--output', default='data/linguistics/typology_prior.json')
    parser.add_argument('--top_k', type=int, default=30,
                        help='빈도 상위 몇 개 n-gram의 discriminativeness 평균')
    parser.add_argument('--use_full_data', action='store_true',
                        help='train split 외 전체 데이터 사용 (기본: train만)')
    args = parser.parse_args()

    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    print("[1/3] 데이터 로딩...")
    if args.use_full_data:
        from data.data_loader import get_loaded_df
        df = get_loaded_df(config).copy()
        df['Name'] = df['Name'].apply(replace_disallowed_chars)
        df = df[df['Name'].str.len() > 0].reset_index(drop=True)
        print(f"  전체 데이터 사용: {len(df)} samples")
    else:
        df = _load_train_split(config)
        print(f"  Train split만 사용: {len(df)} samples")

    print(f"\n[2/3] {df['LanguageGroup'].nunique()}개 언어권 typology 계산 (top_k={args.top_k})...")
    typology = compute_typology_prior(df, top_k=args.top_k)

    print(f"\n[3/3] 저장 및 출력")
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(typology, f, indent=2, ensure_ascii=False)
    print(f"  저장: {args.output}")

    print_typology_table(typology)


if __name__ == '__main__':
    main()
