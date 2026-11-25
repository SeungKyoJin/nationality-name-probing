"""
실험 결과 aggregation — 주말 실험 후 월요일 바로 정리.

Step 0~10에서 저장된 experiment_results.csv를 종합하여
논문 본문 / appendix 테이블 CSV를 자동 생성한다.

Usage:
    python -m analysis.aggregate_results [--exp_dir outputs/exp_MMDD/]

출력:
    outputs/tables/paper_table1_baselines.csv       (Table 1)
    outputs/tables/paper_table2_ablation_hp.csv     (Table 2 — α, channel 등)
    outputs/tables/paper_table3_architecture.csv    (Table 3 — pooling/fusion/contrast)
    outputs/tables/paper_appendix_full.csv          (전체 11-metric)
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

import pandas as pd


# ─── 공통 util ───

CLASSIFICATION_METRICS = [
    'nat_top1', 'nat_top5', 'nat_macro_f1', 'nat_weighted_f1',
    'lang_top1', 'lang_top3', 'lang_top5', 'lang_macro_f1', 'lang_weighted_f1',
]
# CNL metrics (학습 후 별도 기록 필요 — CNL auto-logging hook)
CNL_METRICS = ['cnl_rho_all', 'cnl_rho_subset']

ALL_METRICS = CLASSIFICATION_METRICS + CNL_METRICS


def find_latest_exp_dir(results_root='outputs'):
    """가장 최근 exp_MMDD 디렉토리 반환."""
    candidates = sorted(glob.glob(os.path.join(results_root, 'exp_*')))
    if not candidates:
        raise FileNotFoundError(f"No exp_* directory under {results_root}")
    return candidates[-1]


def load_all_results(exp_dir):
    """exp_dir 하위의 experiment_results.csv 전체를 하나로 통합."""
    csv_paths = glob.glob(os.path.join(exp_dir, '**', 'experiment_results.csv'),
                          recursive=True)
    if not csv_paths:
        # checkpoints 폴더의 누적 CSV도 fallback
        fallback = os.path.join(exp_dir, 'checkpoints', 'experiment_results.csv')
        if os.path.exists(fallback):
            csv_paths = [fallback]

    if not csv_paths:
        raise FileNotFoundError(f"No experiment_results.csv in {exp_dir}")

    dfs = []
    for p in csv_paths:
        try:
            df = pd.read_csv(p)
            df['_source'] = p
            dfs.append(df)
        except Exception as e:
            print(f"[WARN] {p} 읽기 실패: {e}", file=sys.stderr)
    return pd.concat(dfs, ignore_index=True).drop_duplicates()


def select_metrics(df, metric_subset=None):
    """존재하는 metric columns만 뽑기."""
    cols = [c for c in (metric_subset or ALL_METRICS) if c in df.columns]
    meta_cols = [c for c in ['save_name', 'model_type', 'ngram', 'alpha',
                              'fusion', 'pooling', 'contrastive_weight',
                              'augment', 'class_weight', 'max_length']
                 if c in df.columns]
    return df[meta_cols + cols]


# ─── Table 1: Baseline 비교 ───

def build_table1_baselines(df):
    """
    Step 5 (DL baselines) + 5b (LR) + Ours 최종 모델 비교.

    save_name 기준 필터:
      - baseline_cnn, baseline_bilstm, baseline_fasttext, baseline_lr, ours_transformer
    """
    target = ['baseline_lr', 'baseline_fasttext', 'baseline_cnn',
              'baseline_bilstm', 'ours_transformer']
    sub = df[df['save_name'].astype(str).str.contains(
        '|'.join(target), na=False, case=False)].copy()
    if len(sub) == 0:
        print("[WARN] Table 1: baseline 결과 없음")
        return None

    # save_name 별 가장 최근 1개만 (timestamp로)
    if 'timestamp' in sub.columns:
        sub = sub.sort_values('timestamp').drop_duplicates('save_name', keep='last')

    return select_metrics(sub.sort_values('save_name'))


# ─── Table 2: Hyperparameter Ablation (α, channel) ───

def build_table2_hp_ablation(df):
    """
    Step 1 (α) + Step 2 (channel) 결과.
    """
    sub = df[df['save_name'].astype(str).str.contains(
        r'ablation_alpha|ablation_channel', na=False, case=False, regex=True
    )].copy()
    if len(sub) == 0:
        print("[WARN] Table 2: hp ablation 결과 없음")
        return None
    return select_metrics(sub.sort_values('save_name'))


# ─── Table 3: Architecture Ablation (pooling/fusion/contrastive) ───

def build_table3_architecture(df):
    """
    Step 7 (pooling) + Step 8 (fusion) + Step 9 (contrastive) + Step 10 (combined) 결과.
    """
    sub = df[df['save_name'].astype(str).str.contains(
        r'ablation_pool|ablation_fusion|ablation_contrast|combined_best|arch_',
        na=False, case=False, regex=True,
    )].copy()
    if len(sub) == 0:
        print("[WARN] Table 3: architecture ablation 결과 없음")
        return None
    return select_metrics(sub.sort_values('save_name'))


# ─── Appendix Full Table ───

def build_appendix_full(df):
    """전체 실험 결과 그대로 (11-metric 전부)."""
    return select_metrics(df)


# ─── 메인 ───

def main():
    parser = argparse.ArgumentParser(description='실험 결과 aggregation')
    parser.add_argument('--exp_dir', default=None,
                        help='experiment 디렉토리 (미지정 시 가장 최근 exp_MMDD)')
    parser.add_argument('--output_dir', default='outputs/tables',
                        help='출력 CSV 저장 경로')
    args = parser.parse_args()

    exp_dir = args.exp_dir or find_latest_exp_dir()
    print(f"[aggregate] 소스: {exp_dir}")

    df = load_all_results(exp_dir)
    print(f"[aggregate] 총 {len(df)} 실험 행 로드")

    os.makedirs(args.output_dir, exist_ok=True)

    specs = [
        ('paper_table1_baselines.csv', build_table1_baselines,
         'Table 1 (Baseline 비교)'),
        ('paper_table2_ablation_hp.csv', build_table2_hp_ablation,
         'Table 2 (α + channel ablation)'),
        ('paper_table3_architecture.csv', build_table3_architecture,
         'Table 3 (Pooling / Fusion / Contrastive)'),
        ('paper_appendix_full.csv', build_appendix_full,
         'Appendix (11-metric 전체)'),
    ]

    summary = []
    for fname, build_fn, label in specs:
        table = build_fn(df)
        path = os.path.join(args.output_dir, fname)
        if table is None or len(table) == 0:
            summary.append(f"  [skip] {label}: 데이터 부족")
            continue
        table.to_csv(path, index=False)
        summary.append(f"  ✓ {label:40s}  → {path}  ({len(table)}행)")

    print("\n=== Aggregation 완료 ===")
    for line in summary:
        print(line)

    # 요약 텍스트로도 출력 — 월요일 한눈에 확인용
    print("\n=== 주요 수치 요약 (Ours 기준) ===")
    ours_candidates = df[df['save_name'].astype(str).str.contains(
        'ours|combined_best|multi_a0.5', na=False, case=False, regex=True)]
    if len(ours_candidates) > 0:
        ours = ours_candidates.sort_values('timestamp').iloc[-1] if 'timestamp' in ours_candidates.columns else ours_candidates.iloc[-1]
        for m in CLASSIFICATION_METRICS:
            if m in ours.index and pd.notna(ours[m]):
                print(f"  {m:25s} = {ours[m]:.4f}")
    else:
        print("  (ours 식별 불가 — save_name 확인 필요)")


if __name__ == '__main__':
    main()
