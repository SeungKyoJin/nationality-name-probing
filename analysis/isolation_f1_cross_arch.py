"""
analysis/isolation_f1_cross_arch.py

isolation–F1 상관(paper §5, ρ=+0.574 @ Conformer 단일 시드)을
Table 1의 7개 아키텍처 전부로 확장 (rebuttal R2, Reviewer HUuH 대응).

HUuH 지적: "두 번째 발견(isolation ↔ per-class F1)은 한 모델에서만 보고됐다.
첫 번째 발견은 아키텍처 교차 검증을 했으면서."
→ 저장된 class_report_nat CSV(재학습 불필요)로 모델별 ρ를 산출해
  발견이 아키텍처 불변임을 보인다.

방법 (analysis/failure_modes.py와 동일 프로토콜):
  · isolation = self_cnl − max_foreign_cnl (dataset 내 NOC 간 max)
  · 필터: NOC total sample ≥ 500 (cap 3000) → n=112
  · Spearman ρ + 95% pair-resampling bootstrap CI (n_boot=1000)
  · multi-seed 모델(CNN, Conformer)은 시드별 ρ + 시드평균 F1 기준 ρ 둘 다 산출

산출:
  · outputs/analysis/isolation_f1_cross_arch.csv  (report 단위 1행)
  · outputs/analysis/isolation_f1_cross_arch_summary.csv  (모델 단위 집계)

실행:
  python -m analysis.isolation_f1_cross_arch --config configs/base_config.yaml
"""

from __future__ import annotations

import argparse
import os
from functools import reduce

import pandas as pd
import yaml

from analysis.failure_modes import (
    load_class_report, load_dicl, get_noc_total_counts,
    compute_self_and_max_foreign, spearman_with_bootstrap,
)
from analysis.noc_iso3_mapping import noc_to_iso3

CKPT = os.path.join('outputs', 'checkpoints')

# 모델별 canonical class report (paper_draft.tex 주석의 출처 디렉토리 기준).
# multi-seed 모델은 시드 라벨 → 경로, 단일 시드는 {'s42': 경로} 하나.
REPORT_MANIFEST = {
    'LR (TF-IDF)': {
        's42': os.path.join(CKPT, '260516', 'class_report_nat_lr_baseline.csv'),
    },
    'FastText': {
        's42': os.path.join(CKPT, '260513', 'class_report_nat_baseline_fasttext.pt.csv'),
    },
    'BiLSTM': {
        's42': os.path.join(CKPT, '260513', 'class_report_nat_baseline_bilstm.pt.csv'),
    },
    'CNN': {
        's42': os.path.join(CKPT, '260513', 'class_report_nat_baseline_cnn.pt.csv'),
        's7': os.path.join(CKPT, '260516', 'class_report_nat_multiseed_baseline_cnn_seed7.pt.csv'),
        's123': os.path.join(CKPT, '260517', 'class_report_nat_multiseed_baseline_cnn_seed123.pt.csv'),
    },
    'Transformer (MC)': {
        's42': os.path.join(CKPT, '260513', 'class_report_nat_ours_transformer.pt.csv'),
    },
    'CANINE-s': {
        's42': os.path.join('outputs', 'canine_reference', 'class_report_nat_canine_s.csv'),
    },
    'Conformer (ours)': {
        's42': os.path.join(CKPT, '260516', 'class_report_nat_multiseed_ours_alpha03_seed42.pt.csv'),
        's7': os.path.join(CKPT, '260516', 'class_report_nat_multiseed_ours_alpha03_seed7.pt.csv'),
        's123': os.path.join(CKPT, '260517', 'class_report_nat_multiseed_ours_alpha03_seed123.pt.csv'),
    },
    # paper §5 ρ=0.574의 원본 리포트 (단일 시드 conf_alpha_0.3) — 재현 검증용
    'Conformer (paper run)': {
        's42': os.path.join(CKPT, '260514', 'class_report_nat_conf_alpha_0.3.pt.csv'),
    },
}


def build_isolation_table(config, dicl_path, cap, min_samples):
    """NOC별 isolation 지표 + sample count 필터 마스크 (모델 무관, 1회 계산)."""
    raw_path = config['data']['raw_path']
    noc_counts = get_noc_total_counts(raw_path, cap=cap)

    nocs = pd.DataFrame({'NOC': sorted(noc_counts)})
    nocs['n_total'] = nocs['NOC'].map(noc_counts)
    nocs['iso3'] = nocs['NOC'].apply(noc_to_iso3)

    dicl_df = load_dicl(dicl_path)
    cnl_stats = compute_self_and_max_foreign(dicl_df, dataset_iso3=set(nocs['iso3']))

    merged = nocs.merge(cnl_stats, on='iso3', how='left')
    merged[['self_cnl', 'max_foreign_cnl']] = \
        merged[['self_cnl', 'max_foreign_cnl']].fillna(0.0)
    merged['isolation_index'] = merged['self_cnl'] - merged['max_foreign_cnl']
    return merged[merged['n_total'] >= min_samples].copy()


def rho_for_f1(iso_df, f1_df, n_boot, seed):
    """isolation 테이블 × per-NOC F1 → 3개 Spearman(+CI)."""
    merged = iso_df.merge(f1_df, on='NOC', how='inner')
    stats = {}
    for col, key in [('isolation_index', 'rho_isolation'),
                     ('max_foreign_cnl', 'rho_max_foreign'),
                     ('self_cnl', 'rho_self_cnl')]:
        s = spearman_with_bootstrap(
            merged[col].to_numpy(), merged['f1-score'].to_numpy(), n_boot, seed)
        stats[key] = s['rho']
        stats[f'{key}_ci_low'] = s['ci_low']
        stats[f'{key}_ci_high'] = s['ci_high']
        stats[f'{key}_p'] = s['p']
    stats['n_countries'] = len(merged)
    return stats


def seed_mean_f1(report_paths):
    """여러 시드의 class report → NOC별 F1 시드 평균."""
    frames = [
        load_class_report(p)[['NOC', 'f1-score']].rename(
            columns={'f1-score': f'f1_{i}'})
        for i, p in enumerate(report_paths)
    ]
    merged = reduce(lambda a, b: a.merge(b, on='NOC', how='inner'), frames)
    f1_cols = [c for c in merged.columns if c.startswith('f1_')]
    merged['f1-score'] = merged[f1_cols].mean(axis=1)
    return merged[['NOC', 'f1-score']]


def main():
    p = argparse.ArgumentParser(description='Cross-architecture isolation–F1 분석')
    p.add_argument('--config', default=os.path.join('configs', 'base_config.yaml'))
    p.add_argument('--dicl_path',
                   default=os.path.join('data', 'linguistics', 'DICL_v2.csv'))
    p.add_argument('--output_dir', default=os.path.join('outputs', 'analysis'))
    p.add_argument('--n_boot', type=int, default=1000)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--cap', type=int, default=3000)
    p.add_argument('--min_samples', type=int, default=500)
    args = p.parse_args()

    with open(args.config, encoding='utf-8') as f:
        config = yaml.safe_load(f)

    iso_df = build_isolation_table(config, args.dicl_path, args.cap, args.min_samples)
    print(f'[isolation table] {len(iso_df)} NOCs (min_samples ≥ {args.min_samples})')

    per_report_rows, summary_rows = [], []
    for model, seed_paths in REPORT_MANIFEST.items():
        available = {s: p for s, p in seed_paths.items() if os.path.exists(p)}
        missing = set(seed_paths) - set(available)
        if missing:
            print(f'[WARN] {model}: 누락 리포트 {sorted(missing)} → 있는 시드만 사용')
        if not available:
            print(f'[WARN] {model}: 사용 가능한 리포트 없음 → 스킵')
            continue

        # 시드별 ρ
        seed_rhos = []
        for seed_label, path in available.items():
            f1_df = load_class_report(path)[['NOC', 'f1-score']]
            stats = rho_for_f1(iso_df, f1_df, args.n_boot, args.seed)
            per_report_rows.append(
                {'model': model, 'seed': seed_label, 'report': path, **stats})
            seed_rhos.append(stats['rho_isolation'])

        # 시드평균 F1 기준 ρ (multi-seed 모델의 대표값)
        mean_stats = rho_for_f1(
            iso_df, seed_mean_f1(list(available.values())), args.n_boot, args.seed)
        summary_rows.append({
            'model': model,
            'n_seeds': len(available),
            'rho_isolation_seedmean': mean_stats['rho_isolation'],
            'ci_low': mean_stats['rho_isolation_ci_low'],
            'ci_high': mean_stats['rho_isolation_ci_high'],
            'p': mean_stats['rho_isolation_p'],
            'rho_per_seed_min': min(seed_rhos),
            'rho_per_seed_max': max(seed_rhos),
            'rho_max_foreign': mean_stats['rho_max_foreign'],
            'rho_self_cnl': mean_stats['rho_self_cnl'],
            'n_countries': mean_stats['n_countries'],
        })

    os.makedirs(args.output_dir, exist_ok=True)
    per_report_csv = os.path.join(args.output_dir, 'isolation_f1_cross_arch.csv')
    summary_csv = os.path.join(args.output_dir, 'isolation_f1_cross_arch_summary.csv')
    pd.DataFrame(per_report_rows).to_csv(per_report_csv, index=False)
    pd.DataFrame(summary_rows).to_csv(summary_csv, index=False)

    print('\n' + '=' * 78)
    print(' isolation–F1 Spearman ρ — cross-architecture')
    print('=' * 78)
    print(f'{"Model":22s} {"ρ_iso":>7s} {"95% CI":>18s} {"seed range":>14s}'
          f' {"ρ_maxfor":>9s} {"ρ_self":>7s}')
    for r in summary_rows:
        ci = f'[{r["ci_low"]:+.3f}, {r["ci_high"]:+.3f}]'
        seed_range = (f'{r["rho_per_seed_min"]:+.3f}~{r["rho_per_seed_max"]:+.3f}'
                      if r['n_seeds'] > 1 else '—')
        print(f'{r["model"]:22s} {r["rho_isolation_seedmean"]:+7.3f} {ci:>18s}'
              f' {seed_range:>14s} {r["rho_max_foreign"]:+9.3f}'
              f' {r["rho_self_cnl"]:+7.3f}')
    print(f'\n저장: {per_report_csv}\n저장: {summary_csv}')


if __name__ == '__main__':
    main()
