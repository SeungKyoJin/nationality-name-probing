# -*- coding: utf-8 -*-
"""모델 × DICL 전 지표(11개) Spearman ρ 매트릭스 생성.

메타리뷰 요구 "compare all of the methods to all of the facets of DICL" 대응.
학습/추론 없이, rebuttal 때 저장해 둔 모델별 pair similarity CSV
(outputs/analysis/table1_rho_ci/pairs_*.csv, outputs/canine_reference/pair_similarities.csv)
를 DICL_v2.csv와 머지해서 상관만 재계산한다.

출력:
  outputs/analysis/dicl_facet_matrix.csv        (long format: model, metric, scope, rho, n)
  outputs/analysis/dicl_facet_matrix_wide.csv   (paper Table 1용 wide format)

사용법:
  python -m analysis.dicl_facet_matrix
"""

import os
import sys

import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.noc_iso3_mapping import noc_to_iso3

DICL_PATH = 'data/linguistics/DICL_v2.csv'
PAIRS_DIR = 'outputs/analysis/table1_rho_ci'
CANINE_PAIRS = 'outputs/canine_reference/pair_similarities.csv'
OUT_DIR = 'outputs/analysis'

# DICL 11개 지표 전부 (README_DICL_v2.txt 순서)
DICL_METRICS = ['col', 'cor', 'cnl', 'cal', 'csl',
                'lpn', 'lpa', 'lps', 'bpn', 'bpa', 'bps']

# Table 1 표기 순서와 일치시키기 위한 모델 목록 (파일명, 표기명)
MODELS = [
    ('pairs_lr.csv',          'LR (TF-IDF)'),
    ('pairs_fasttext.csv',    'FastText'),
    ('pairs_bilstm.csv',      'BiLSTM'),
    ('pairs_cnn.csv',         'CNN'),
    ('pairs_transformer.csv', 'Transformer'),
    ('pairs_vanilla_mcc.csv', 'Vanilla MCC'),
    (CANINE_PAIRS,            'CANINE-s'),
    ('pairs_ours.csv',        'Conformer'),   # 구 "Ours" — EACL 개정판에서 리네이밍
]


def _pair_key(iso_a, iso_b):
    """무순서 페어 키. DICL은 mirrored라서 정렬 튜플로 통일한다."""
    return tuple(sorted((iso_a, iso_b)))


def load_dicl_full(dicl_path=DICL_PATH):
    """DICL 전 지표를 무순서 페어 키로 인덱싱한 DataFrame 반환."""
    dicl = pd.read_csv(dicl_path)
    dicl['key'] = [_pair_key(i, j) for i, j in zip(dicl['iso3_i'], dicl['iso3_j'])]
    # mirrored 중복 제거 (같은 무순서 페어 2행 → 1행)
    dicl = dicl.drop_duplicates(subset='key').set_index('key')
    return dicl[DICL_METRICS]


def load_model_pairs(path):
    """모델 pair similarity CSV → 무순서 iso3 페어 키 + emb_sim."""
    pairs = pd.read_csv(path)
    keys, sims, cnl_saved = [], [], []
    for noc_i, noc_j, sim, cnl in zip(pairs['noc_i'], pairs['noc_j'],
                                      pairs['emb_sim'], pairs['cnl']):
        iso_i, iso_j = noc_to_iso3(noc_i), noc_to_iso3(noc_j)
        if iso_i is None or iso_j is None:
            continue
        keys.append(_pair_key(iso_i, iso_j))
        sims.append(sim)
        cnl_saved.append(cnl)
    return pd.DataFrame({'key': keys, 'emb_sim': sims,
                         'cnl_saved': cnl_saved}).set_index('key')


def compute_matrix():
    dicl = load_dicl_full()
    rows = []
    for fname, label in MODELS:
        path = fname if os.path.sep in fname or '/' in fname \
            else os.path.join(PAIRS_DIR, fname)
        model_pairs = load_model_pairs(path)
        merged = model_pairs.join(dicl, how='inner')

        # sanity check: 저장 당시 cnl과 DICL 머지 cnl이 일치해야 함
        chk = merged.dropna(subset=['cnl'])
        max_dev = (chk['cnl'] - chk['cnl_saved']).abs().max()
        if max_dev > 1e-6:
            raise ValueError(f'{label}: cnl mismatch (max dev {max_dev})')

        for metric in DICL_METRICS:
            valid = merged.dropna(subset=[metric])
            rho, p = spearmanr(valid['emb_sim'], valid[metric])
            rows.append({'model': label, 'metric': metric, 'scope': 'all',
                         'rho': round(rho, 3), 'p_value': p, 'n_pairs': len(valid)})

        # 논문 기존 관례 유지: cnl > 0 subset ρ도 함께 기록
        sub = merged[merged['cnl'] > 0].dropna(subset=['cnl'])
        rho, p = spearmanr(sub['emb_sim'], sub['cnl'])
        rows.append({'model': label, 'metric': 'cnl', 'scope': 'subset_cnl_gt0',
                     'rho': round(rho, 3), 'p_value': p, 'n_pairs': len(sub)})
        print(f'[OK] {label}: {len(merged)} pairs merged')

    long_df = pd.DataFrame(rows)
    long_df.to_csv(os.path.join(OUT_DIR, 'dicl_facet_matrix.csv'), index=False)

    wide = long_df[long_df['scope'] == 'all'].pivot(
        index='model', columns='metric', values='rho')[DICL_METRICS]
    sub_col = long_df[long_df['scope'] == 'subset_cnl_gt0'].set_index('model')['rho']
    wide.insert(loc=DICL_METRICS.index('cnl') + 1, column='cnl_sub', value=sub_col)
    # Table 1 행 순서 유지
    wide = wide.reindex([label for _, label in MODELS])
    wide.to_csv(os.path.join(OUT_DIR, 'dicl_facet_matrix_wide.csv'))
    print('\n=== Model x DICL facet matrix (rho, all pairs) ===')
    print(wide.to_string())
    return wide


if __name__ == '__main__':
    compute_matrix()
