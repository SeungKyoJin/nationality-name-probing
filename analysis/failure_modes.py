"""
analysis/failure_modes.py

Per-country classification F1 ↔ DICL linguistic-isolation index 분석.

paper §5 "Failure modes and the classification ceiling" + Table 3 산출.

핵심 정의:
  · self_cnl    = DICL CNL_{ii}     (한 국가 내 native-language homogeneity)
  · max_foreign = max_{j≠i, j ∈ dataset} CNL_{ij}
  · isolation   = self_cnl − max_foreign
  · 3-way partition (paper Table 3 caption):
      Isolate       : self_cnl ≥ θ_self  AND  max_foreign < θ_iso
      Absorbed      : self_cnl ≥ θ_self  AND  max_foreign ≥ θ_abs
      Heterogeneous : self_cnl < θ_self

분석 필터:
  · NOC sample count ≥ --min_samples  (default 500)

산출:
  · failure_modes_per_country.csv
      NOC, iso3, n_total, f1, support, self_cnl, max_foreign_cnl,
      isolation_index, partition
  · failure_modes_summary.json
      - 3 Spearman ρ + 95% pair-resampling bootstrap CI
        · F1 ↔ isolation_index
        · F1 ↔ max_foreign_cnl
        · F1 ↔ self_cnl
      - per-partition: n, mean F1, median F1

Usage:
  python -m analysis.failure_modes \\
      --config configs/base_config.yaml \\
      --class_report outputs/checkpoints/260514/class_report_nat_conf_alpha_0.3.pt.csv \\
      --output_dir outputs/analysis/failure_modes \\
      --n_boot 1000

  # threshold 변경 (paper 기본값과 다르게)
  python -m analysis.failure_modes ... \\
      --self_cnl_threshold 0.7 \\
      --absorbed_max_threshold 0.5 \\
      --isolate_max_threshold 0.3 \\
      --min_samples 500
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict

import numpy as np
import pandas as pd
import yaml
from scipy.stats import spearmanr

from analysis.noc_iso3_mapping import noc_to_iso3


# --------------------------------------------------------------------------
# 데이터 로딩
# --------------------------------------------------------------------------

# sklearn classification_report가 CSV로 저장될 때 끝에 자동 추가하는 요약 행.
_REPORT_FOOTER = {'accuracy', 'macro avg', 'weighted avg'}


def load_class_report(path: str) -> pd.DataFrame:
    """class_report_nat CSV → NOC별 f1-score / support."""
    df = pd.read_csv(path, index_col=0)
    df = df.loc[~df.index.isin(_REPORT_FOOTER)].copy()
    df.index.name = 'NOC'
    return df[['f1-score', 'support']].reset_index()


def load_dicl(dicl_path: str) -> pd.DataFrame:
    """DICL CSV → (iso3_i, iso3_j, cnl)만 추출."""
    return pd.read_csv(dicl_path, usecols=['iso3_i', 'iso3_j', 'cnl'])


def get_noc_total_counts(raw_csv_path: str, cap: int) -> Dict[str, int]:
    """raw 데이터에서 NOC별 total sample count (cap 적용)."""
    df = pd.read_csv(raw_csv_path, usecols=['Name', 'NOC'])
    df = df.dropna(subset=['Name', 'NOC'])
    counts = df.groupby('NOC').size().clip(upper=cap)
    return counts.to_dict()


# --------------------------------------------------------------------------
# DICL 지표 계산
# --------------------------------------------------------------------------

def compute_self_and_max_foreign(
    dicl_df: pd.DataFrame,
    dataset_iso3: set,
) -> pd.DataFrame:
    """
    각 ISO3별 self_cnl(=CNL_ii) 와 max_foreign_cnl 산출.

    max_foreign_cnl은 paper line 687-689 정의에 따라
    데이터셋에 포함된 NOC들 사이에서만 max를 취한다.
    """
    self_pairs = (
        dicl_df[dicl_df['iso3_i'] == dicl_df['iso3_j']]
        .rename(columns={'iso3_i': 'iso3', 'cnl': 'self_cnl'})
        [['iso3', 'self_cnl']]
    )

    foreign = dicl_df[
        (dicl_df['iso3_i'] != dicl_df['iso3_j']) &
        (dicl_df['iso3_i'].isin(dataset_iso3)) &
        (dicl_df['iso3_j'].isin(dataset_iso3))
    ]
    max_for = (
        foreign.groupby('iso3_i')['cnl'].max()
        .reset_index()
        .rename(columns={'iso3_i': 'iso3', 'cnl': 'max_foreign_cnl'})
    )
    return (
        self_pairs.merge(max_for, on='iso3', how='outer').fillna(0.0)
    )


def partition_country(
    self_cnl: float,
    max_for: float,
    self_thr: float,
    abs_thr: float,
    iso_thr: float,
) -> str:
    """paper Table 3 caption의 3-way 분류 (+ 'Other'는 미분류 케이스)."""
    if self_cnl < self_thr:
        return 'Heterogeneous'
    if max_for < iso_thr:
        return 'Isolate'
    if max_for >= abs_thr:
        return 'Absorbed'
    return 'Other'  # self_cnl≥θ_self AND θ_iso ≤ max_for < θ_abs


# --------------------------------------------------------------------------
# 통계: Spearman ρ + bootstrap CI
# --------------------------------------------------------------------------

def spearman_with_bootstrap(
    x: np.ndarray,
    y: np.ndarray,
    n_boot: int,
    seed: int = 42,
) -> dict:
    """Spearman ρ + 95% pair-resampling bootstrap CI."""
    rho, p = spearmanr(x, y)
    rng = np.random.default_rng(seed)
    n = len(x)
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        r, _ = spearmanr(x[idx], y[idx])
        if not np.isnan(r):
            boots.append(r)
    boots = np.asarray(boots)
    ci_lo, ci_hi = np.percentile(boots, [2.5, 97.5])
    return {
        'rho': float(rho),
        'p': float(p),
        'ci_low': float(ci_lo),
        'ci_high': float(ci_hi),
        'n_boot_valid': int(len(boots)),
        'n': int(n),
    }


# --------------------------------------------------------------------------
# 메인 파이프라인
# --------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        description='Failure-mode partition + isolation index analysis',
    )
    p.add_argument('--config', default=os.path.join('configs', 'base_config.yaml'))
    p.add_argument('--class_report', required=True,
                   help='class_report_nat_*.pt.csv (Trainer.evaluate 산출)')
    p.add_argument('--dicl_path',
                   default=os.path.join('data', 'linguistics', 'DICL_v2.csv'))
    p.add_argument('--output_dir',
                   default=os.path.join('outputs', 'analysis', 'failure_modes'))
    p.add_argument('--n_boot', type=int, default=1000)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--cap', type=int, default=3000,
                   help='per-NOC sample cap (raw 카운트 clipping 용)')
    p.add_argument('--min_samples', type=int, default=500,
                   help='분석 대상 NOC의 최소 total sample 수')
    p.add_argument('--self_cnl_threshold', type=float, default=0.7)
    p.add_argument('--absorbed_max_threshold', type=float, default=0.5)
    p.add_argument('--isolate_max_threshold', type=float, default=0.3)
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    with open(args.config, encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # 1) per-NOC F1
    f1_df = load_class_report(args.class_report)

    # 2) per-NOC total sample count (raw 데이터 기준, cap 적용)
    raw_path = config['data']['raw_path']
    noc_counts = get_noc_total_counts(raw_path, cap=args.cap)
    f1_df['n_total'] = f1_df['NOC'].map(noc_counts).fillna(0).astype(int)

    # 3) NOC → ISO3
    f1_df['iso3'] = f1_df['NOC'].apply(noc_to_iso3)

    # 4) DICL 로딩
    dicl_df = load_dicl(args.dicl_path)

    # 5) self_cnl + max_foreign_cnl
    cnl_stats = compute_self_and_max_foreign(
        dicl_df, dataset_iso3=set(f1_df['iso3'])
    )

    # 6) merge + isolation_index
    merged = f1_df.merge(cnl_stats, on='iso3', how='left')
    merged[['self_cnl', 'max_foreign_cnl']] = \
        merged[['self_cnl', 'max_foreign_cnl']].fillna(0.0)
    merged['isolation_index'] = (
        merged['self_cnl'] - merged['max_foreign_cnl']
    )

    # 7) sample-count 필터
    keep = merged[merged['n_total'] >= args.min_samples].copy()

    # 8) 3-way partition
    keep['partition'] = keep.apply(
        lambda r: partition_country(
            r['self_cnl'], r['max_foreign_cnl'],
            args.self_cnl_threshold,
            args.absorbed_max_threshold,
            args.isolate_max_threshold,
        ),
        axis=1,
    )

    # 9) per-country CSV
    out_csv = os.path.join(args.output_dir, 'failure_modes_per_country.csv')
    (
        keep[['NOC', 'iso3', 'n_total', 'f1-score', 'support',
              'self_cnl', 'max_foreign_cnl', 'isolation_index', 'partition']]
        .sort_values('f1-score', ascending=False)
        .to_csv(out_csv, index=False)
    )

    # 10) Spearman + bootstrap
    f1 = keep['f1-score'].to_numpy()
    iso = keep['isolation_index'].to_numpy()
    self_c = keep['self_cnl'].to_numpy()
    max_f = keep['max_foreign_cnl'].to_numpy()

    result = {
        'n_countries': int(len(keep)),
        'min_samples': int(args.min_samples),
        'cap': int(args.cap),
        'thresholds': {
            'self_cnl': args.self_cnl_threshold,
            'absorbed_max': args.absorbed_max_threshold,
            'isolate_max': args.isolate_max_threshold,
        },
        'spearman': {
            'f1_vs_isolation': spearman_with_bootstrap(
                iso, f1, args.n_boot, args.seed),
            'f1_vs_max_foreign': spearman_with_bootstrap(
                max_f, f1, args.n_boot, args.seed),
            'f1_vs_self_cnl': spearman_with_bootstrap(
                self_c, f1, args.n_boot, args.seed),
        },
    }

    # 11) per-partition summary
    partition_summary = (
        keep.groupby('partition')
        .agg(n=('NOC', 'count'),
             mean_f1=('f1-score', 'mean'),
             median_f1=('f1-score', 'median'),
             examples=('NOC', lambda s: list(s.iloc[:6])))
        .reset_index()
    )
    # JSON 직렬화 가능한 dict로
    result['partition_summary'] = [
        {
            'partition': row.partition,
            'n': int(row.n),
            'mean_f1': float(row.mean_f1),
            'median_f1': float(row.median_f1),
            'examples': list(row.examples),
        }
        for row in partition_summary.itertuples(index=False)
    ]

    out_json = os.path.join(args.output_dir, 'failure_modes_summary.json')
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    # 12) Console 요약
    print('\n' + '=' * 64)
    print(' Failure-mode analysis')
    print('=' * 64)
    print(f'  per-country CSV : {out_csv}')
    print(f'  summary JSON    : {out_json}')
    print(f'  n_countries     : {len(keep)}'
          f' (min_samples ≥ {args.min_samples})')

    print('\n  Spearman ρ (95% bootstrap CI, n_boot=' + str(args.n_boot) + ')')
    for key, label in [
        ('f1_vs_isolation',   'F1 ↔ isolation_index'),
        ('f1_vs_max_foreign', 'F1 ↔ max_foreign_cnl'),
        ('f1_vs_self_cnl',    'F1 ↔ self_cnl'),
    ]:
        s = result['spearman'][key]
        sign = '+' if s['rho'] >= 0 else ''
        print(f'    {label:30s}: {sign}{s["rho"]:.3f}'
              f'  [{s["ci_low"]:+.3f}, {s["ci_high"]:+.3f}]'
              f'  p={s["p"]:.2e}')

    print('\n  Partition (mean F1)')
    for row in result['partition_summary']:
        examples = ', '.join(row['examples'])
        print(f'    {row["partition"]:15s} n={row["n"]:3d}'
              f'  mean F1={row["mean_f1"]:.3f}'
              f'  examples: {examples}')


if __name__ == '__main__':
    main()
