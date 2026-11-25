"""
LR baseline의 ρ vs DICL CNL 계산.

LR은 임베딩을 만들지 않지만 학습된 coefficient (lr.coef_) 가
국가별 feature 가중치 vector → "country representation" 으로 사용 가능.

이 벡터들의 pairwise cosine similarity를 DICL CNL과 비교 → ρ.

Usage:
    python -m analysis.dicl_lr --config configs/base_config.yaml
"""

import argparse
import os
from datetime import datetime

import numpy as np
import pandas as pd
import yaml
from scipy.stats import spearmanr
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity

from utils.util import set_seed, replace_disallowed_chars


def load_train_split(config):
    """run_baseline_lr.py와 동일한 split 로직으로 train + test 반환."""
    df = pd.read_csv(config['data']['raw_path'])
    df = df.dropna(subset=['Name', 'NOC', 'LanguageGroup']).copy()

    df['Name'] = df['Name'].apply(replace_disallowed_chars)
    df = df[df['Name'].str.len() > 0].reset_index(drop=True)

    # Stratified split (data_loader.robust_split 동일)
    train_idx = []
    for _, group in df.groupby('NOC'):
        idxs = group.index.values.copy()
        np.random.shuffle(idxs)
        n = len(idxs)
        if n >= 3:
            cut = max(1, int(0.8 * n))
            train_idx.extend(idxs[:cut])
        else:
            train_idx.append(idxs[0])
    return df.iloc[train_idx].reset_index(drop=True)


def load_dicl_cnl(dicl_path='data/linguistics/DICL_v2.csv'):
    """
    DICL CNL pairwise table 로드 (iso3 기준).

    DICL 원본은 (A,B), (B,A) 양방향 row가 모두 있어서 그대로 쓰면 중복.
    deep learning 분석(`analysis.dicl_correlation.load_dicl`)과 동일하게
    sorted tuple 기준으로 dedup 후 long-form DataFrame 반환.
    """
    df = pd.read_csv(dicl_path)
    cnl_col = next(c for c in df.columns if c.lower() == 'cnl')
    df = df[['iso3_i', 'iso3_j', cnl_col]].rename(columns={cnl_col: 'cnl'})
    df = df[df['iso3_i'] != df['iso3_j']].copy()                # 자기쌍 제거
    df['_key'] = df.apply(
        lambda r: tuple(sorted((r['iso3_i'], r['iso3_j']))), axis=1)
    df = df.drop_duplicates(subset='_key').drop(columns='_key')   # (A,B)==(B,A) dedup
    return df


def compute_cnl_rho_lr(lr_model, dicl_path='data/linguistics/DICL_v2.csv'):
    """
    학습된 LR 모델 객체로부터 ρ vs DICL CNL 계산.

    Args:
        lr_model: 학습된 sklearn LogisticRegression. lr.coef_ + lr.classes_ 사용.
        dicl_path: DICL_v2.csv 경로 (iso3_i / iso3_j / cnl 컬럼 사용).

    Returns:
        dict {'cnl_rho_all', 'cnl_rho_subset', 'cnl_n_pairs'} — 학습 시 자동 로그용.
        실패 시 None.
    """
    try:
        from analysis.noc_iso3_mapping import noc_to_iso3

        countries = list(lr_model.classes_)            # NOC codes
        coef = lr_model.coef_
        sim_matrix = cosine_similarity(coef)

        # NOC → ISO3로 매핑 (DICL은 iso3_i / iso3_j 형식)
        iso_to_idx = {}
        for i, noc in enumerate(countries):
            iso = noc_to_iso3(noc)
            if iso:
                iso_to_idx[iso] = i

        dicl = load_dicl_cnl(dicl_path)
        pair_sim, pair_cnl = [], []
        for _, row in dicl.iterrows():
            i, j = row['iso3_i'], row['iso3_j']
            if i in iso_to_idx and j in iso_to_idx:
                pair_sim.append(sim_matrix[iso_to_idx[i], iso_to_idx[j]])
                pair_cnl.append(row['cnl'])
        pair_sim = np.array(pair_sim)
        pair_cnl = np.array(pair_cnl)
        if len(pair_sim) == 0:
            print("[compute_cnl_rho_lr] DICL과 매칭되는 country pair 0개 — NOC↔ISO3 매핑 확인")
            return None

        rho_all, _ = spearmanr(pair_sim, pair_cnl)
        sub_mask = pair_cnl > 0
        rho_sub = float('nan')
        if sub_mask.sum() >= 10:
            rho_sub, _ = spearmanr(pair_sim[sub_mask], pair_cnl[sub_mask])

        return {
            'cnl_rho_all': round(float(rho_all), 4),
            'cnl_rho_subset': round(float(rho_sub), 4),
            'cnl_n_pairs': int(len(pair_sim)),
        }
    except Exception as e:
        print(f"[compute_cnl_rho_lr] 실패: {e}")
        return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', default='configs/base_config.yaml')
    p.add_argument('--output_csv', default=None,
                   help='기본: outputs/checkpoints/<date>/dicl_lr_baseline.csv (딥러닝 실험과 동일)')
    args = p.parse_args()

    config = yaml.safe_load(open(args.config, 'r', encoding='utf-8'))
    set_seed(config['experiment']['seed'])

    # 출력 경로 — 딥러닝 실험과 동일 구조
    if args.output_csv is None:
        date_dir = datetime.now().strftime("%y%m%d")
        output_dir = os.path.join(
            config['experiment']['result_directory'],
            config['train']['save_dir'],
            date_dir,
        )
        args.output_csv = os.path.join(output_dir, 'dicl_lr_baseline.csv')

    print("[1/4] 데이터 로딩 + train split")
    train_df = load_train_split(config)
    print(f"  train: {len(train_df)} names, {train_df['NOC'].nunique()} countries")

    print("[2/4] TF-IDF + LR 학습")
    vec = TfidfVectorizer(analyzer='char_wb', ngram_range=(1, 3),
                          min_df=2, max_features=20000, lowercase=True)
    X = vec.fit_transform(train_df['Name'])
    lr = LogisticRegression(max_iter=1000, C=1.0, solver='saga', n_jobs=-1)
    lr.fit(X, train_df['NOC'])
    print(f"  coef shape: {lr.coef_.shape}")

    print("[3/4] LR coefficient → pairwise cosine similarity")
    from analysis.noc_iso3_mapping import noc_to_iso3
    countries = list(lr.classes_)                       # NOC codes
    coef = lr.coef_                                     # [n_countries, n_features]
    sim_matrix = cosine_similarity(coef)                # [n_countries, n_countries]

    # NOC → ISO3 → row idx (DICL은 iso3 컬럼 사용)
    iso_to_idx = {}
    for i, noc in enumerate(countries):
        iso = noc_to_iso3(noc)
        if iso:
            iso_to_idx[iso] = i

    print("[4/4] DICL CNL과 비교 → ρ")
    dicl = load_dicl_cnl()
    pair_sim = []
    pair_cnl = []
    for _, row in dicl.iterrows():
        i, j = row['iso3_i'], row['iso3_j']
        if i in iso_to_idx and j in iso_to_idx:
            pair_sim.append(sim_matrix[iso_to_idx[i], iso_to_idx[j]])
            pair_cnl.append(row['cnl'])
    pair_sim = np.array(pair_sim)
    pair_cnl = np.array(pair_cnl)

    rho_all, p_all = spearmanr(pair_sim, pair_cnl)
    sub_mask = pair_cnl > 0
    rho_sub, p_sub = spearmanr(pair_sim[sub_mask], pair_cnl[sub_mask])

    n_all = len(pair_sim)
    n_sub = sub_mask.sum()
    print()
    print(f"  ρ_all     = {rho_all:+.4f}  (n={n_all},  p={p_all:.2e})")
    print(f"  ρ_subset  = {rho_sub:+.4f}  (n={n_sub},  p={p_sub:.2e})")

    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    pd.DataFrame([
        {'metric': 'rho_all',    'rho': rho_all, 'p': p_all, 'n': n_all},
        {'metric': 'rho_subset', 'rho': rho_sub, 'p': p_sub, 'n': int(n_sub)},
    ]).to_csv(args.output_csv, index=False)
    print(f"\n  저장: {args.output_csv}")


if __name__ == '__main__':
    main()
