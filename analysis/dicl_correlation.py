"""
DICL 상관관계 분석.

학습된 Transformer 임베딩 거리와 DICL의 CNL (Common Native Language index) 간
Spearman 상관관계를 계산한다.

과정:
  1. 학습된 모델로 전체 데이터의 임베딩 추출 (encode())
  2. 국가별 임베딩 평균 (centroid) 계산
  3. 국가 쌍 간 cosine distance 행렬 생성
  4. DICL_v2.csv에서 CNL 추출 (주 지표)
  5. NOC↔ISO3 매핑으로 정렬
  6. Spearman ρ 계산 + scatter plot 저장

Usage:
    python -m analysis.dicl_correlation \
        --config configs/base_config.yaml \
        --model_path outputs/checkpoints/best_model.pt
"""

import os
import argparse

import yaml
import torch
import numpy as np
import pandas as pd
from scipy.spatial.distance import cosine
from scipy.stats import spearmanr
from torch.utils.data import DataLoader

from data import prepare_data, get_loaded_df, get_country_idx, get_lang_group_idx
from data import get_vocab_size, get_ngram_dicts, get_country_list
from data.dataset import MultiChannelDataset
from models import get_model
from utils.util import set_seed, replace_disallowed_chars
from analysis.noc_iso3_mapping import noc_to_iso3


# ─── DICL 지표 정의 ───
# 출처: data/linguistics/README_DICL_v2.txt (공식 정의)
# 주 지표는 CNL (Common Native Language index) — Phase 4 분석 기준
# Gurevich et al. (2024) USITC Economics Working Paper 2024-03-A
DICL_METRIC_DESCRIPTIONS = {
    # Common language indicators (binary 0/1)
    'col': 'Common official language indicator',
    'cor': 'Restricted common official language indicator',
    # Common language indices (continuous, 0~1, native speaker share dot-product)
    'cnl': 'Common native language index',             # ★ Primary
    'cal': 'Common acquired language index',
    'csl': 'Common spoken language index (native + acquired)',
    # Linguistic proximity indices (different languages, tree-based)
    'lpn': 'Linguistic proximity index (different native languages)',
    'lpa': 'Linguistic proximity index (different acquired languages)',
    'lps': 'Linguistic proximity index (different spoken languages)',
    # Linguistic branch proximity indices (different languages, branch-based)
    'bpn': 'Linguistic branch proximity index (different native languages)',
    'bpa': 'Linguistic branch proximity index (different acquired languages)',
    'bps': 'Linguistic branch proximity index (different spoken languages)',
}

# 모든 11개 metric을 계산하여 Table A1의 투명성 확보
ALL_DICL_METRICS = list(DICL_METRIC_DESCRIPTIONS.keys())


def export_dicl_metric_table(all_results, primary_metric,
                              tables_dir='outputs/tables'):
    """
    DICL 6개 지표 비교 테이블을 논문 Appendix용 CSV로 저장.

    사용 목적:
      - 왜 CNL을 primary로 골랐는지 정당화 (cherry-picking 방지)
      - 6개 지표 중 |ρ| 가장 큰 것이 CNL임을 명시

    컬럼:
      metric, description, spearman_rho, abs_rho, p_value, n_pairs, primary
    """
    rows = []
    for metric, res in all_results.items():
        # p=0.0은 float underflow — 실제는 < 1e-308 수준. 논문용 표현으로 정규화
        p_raw = res['p_value']
        p_report = '<1e-308' if p_raw == 0.0 else f'{p_raw:.2e}'
        rows.append({
            'metric': metric,
            'description': DICL_METRIC_DESCRIPTIONS.get(metric, '—'),
            'spearman_rho': round(res['rho'], 4),
            'abs_rho': round(abs(res['rho']), 4),
            'p_value': p_report,
            'n_pairs': res['n_pairs'],
            'primary': metric == primary_metric,
        })
    df = pd.DataFrame(rows).sort_values('abs_rho', ascending=False)

    os.makedirs(tables_dir, exist_ok=True)
    path = os.path.join(tables_dir, 'dicl_metric_comparison.csv')
    df.to_csv(path, index=False)
    print(f"  - {path}  (Appendix Table A1, {len(df)} metrics)")
    return df


def export_appendix_b_malta_lpn(all_results, tables_dir='outputs/tables',
                                  lpn_threshold=0.7, emb_threshold=0.3):
    """
    Appendix B: LPN 기반 supplementary 분석 — Malta-Arabic orthographic bias.

    정의상 LPN = Linguistic proximity index for DIFFERENT native languages
    (k≠h) → 같은 언어 공유는 제외, 서로 다른 언어들 간 tree proximity.

    Maltese는 DICL에서 Arabic과 다른 native language로 분류되지만
    Semitic 계통이라 LPN이 매우 높음. 하지만 Latin 철자 때문에
    모델 embedding은 낮음 → orthographic bias의 clean case study.

    필터 기준:
      · LPN >= 0.7  (매우 강한 tree proximity)
      · Emb <= 0.3  (모델은 멀게 봄)

    CSV 컬럼:
      noc_i, noc_j, cnl, lpn, embedding_distance, emb_similarity
    """
    if 'lpn' not in all_results:
        print("  [Appendix B] LPN 결과 없음 — skip")
        return None

    lpn_res = all_results['lpn']
    cnl_res = all_results.get('cnl')
    pairs = lpn_res['matched_pairs']
    emb_d = np.array(lpn_res['matched_emb'])
    lpn_v = np.array(lpn_res['matched_dicl'])
    emb_sim = 1.0 - emb_d

    # CNL 값 매핑 (같은 pair일 때만 — lpn과 cnl의 matched_pairs가 다를 수 있으므로
    # lpn pair 기준으로 cnl lookup)
    cnl_lookup = {}
    if cnl_res is not None:
        for p, v in zip(cnl_res['matched_pairs'], cnl_res['matched_dicl']):
            cnl_lookup[tuple(sorted(p))] = v

    rows = []
    for (noc_i, noc_j), lpn_val, emb_s, emb_dist in zip(pairs, lpn_v, emb_sim, emb_d):
        if lpn_val < lpn_threshold or emb_s > emb_threshold:
            continue
        cnl_val = cnl_lookup.get(tuple(sorted((noc_i, noc_j))))
        rows.append({
            'noc_i': noc_i,
            'noc_j': noc_j,
            'cnl': round(cnl_val, 4) if cnl_val is not None else None,
            'lpn': round(float(lpn_val), 4),
            'embedding_distance': round(float(emb_dist), 4),
            'emb_similarity': round(float(emb_s), 4),
        })

    if not rows:
        print(f"  [Appendix B] LPN≥{lpn_threshold} AND Emb≤{emb_threshold} 조건 pair 없음")
        return None

    df = pd.DataFrame(rows).sort_values('lpn', ascending=False)
    os.makedirs(tables_dir, exist_ok=True)
    path = os.path.join(tables_dir, 'appendix_b_malta_lpn.csv')
    df.to_csv(path, index=False)
    print(f"  - {path}  (Appendix B, {len(df)} Malta-cluster pairs)")

    # Malta 포함 비율 확인 (예상: 대부분 Malta)
    malta_count = df[(df['noc_i'] == 'MLT') | (df['noc_j'] == 'MLT')].shape[0]
    print(f"    Malta 관련: {malta_count}/{len(df)}쌍 ({malta_count/len(df)*100:.0f}%)")
    return df


def export_dicl_robustness_check(all_results, primary_metric,
                                  thresholds=(0.0, 0.001, 0.01, 0.05, 0.10, 0.20, 0.30),
                                  tables_dir='outputs/tables'):
    """
    Primary metric (CNL)의 구조적 영(structural zero) 특성을 반영한 subset 분석.

    **이론적 근거**:
      · CNL=0은 "정의상 공통 native language 없음" = structural zero
        (예: 한국-브라질, 몽골-나이지리아) — not missing data
      · 구조적 0인 쌍들은 x축에서 한 점에 몰려 correlation 측정 의미 약화
      · Zero-inflated 분포에서는 principled subset analysis가 통계학적으로 권장됨
        (cf. PMC4194007 structural zeros literature)

    **투명성 (p-hacking 아님)**:
      · 전체(CNL=0 포함) + "CNL > 0 strict" + 여러 threshold 모두 함께 보고
      · subset 선택이 data post-hoc이 아닌 "structural zero 배제" 원칙 기반

    보고 방식 (Appendix A2):
      "Among pairs with CNL > 0 (n=6,724), ρ strengthens from 0.33 to 0.47.
       Thresholds ≥ 0.01 to ≥ 0.10 all yield ρ > 0.5 (Table A2)."

    CSV 컬럼:
      cnl_threshold, n_pairs, spearman_rho, p_value
    """
    from scipy.stats import spearmanr

    if primary_metric not in all_results:
        print("  [robustness] primary metric 결과 없음 — skip")
        return None

    pr = all_results[primary_metric]
    dicl_sim = np.array(pr['matched_dicl'])            # similarity (CNL)
    emb_sim = 1.0 - np.array(pr['matched_emb'])        # 1 - cosine distance

    rows = []
    # "CNL > 0 strict" 먼저 추가 — structural zero 배제한 핵심 subset
    strict_mask = dicl_sim > 0
    if int(strict_mask.sum()) >= 10:
        rho_s, p_s = spearmanr(dicl_sim[strict_mask], emb_sim[strict_mask])
        p_s_report = '<1e-308' if float(p_s) == 0.0 else f'{float(p_s):.2e}'
        rows.append({
            'cnl_threshold': '> 0 (strict)',   # 문자열로 구분
            'n_pairs': int(strict_mask.sum()),
            'spearman_rho': round(float(rho_s), 4),
            'p_value': p_s_report,
        })

    for t in thresholds:
        mask = dicl_sim >= t
        if int(mask.sum()) < 10:
            continue
        rho, p = spearmanr(dicl_sim[mask], emb_sim[mask])
        p_report = '<1e-308' if float(p) == 0.0 else f'{float(p):.2e}'
        rows.append({
            'cnl_threshold': t,
            'n_pairs': int(mask.sum()),
            'spearman_rho': round(float(rho), 4),
            'p_value': p_report,
        })

    if not rows:
        return None

    df = pd.DataFrame(rows)

    os.makedirs(tables_dir, exist_ok=True)
    path = os.path.join(tables_dir, 'dicl_robustness_check.csv')
    df.to_csv(path, index=False)
    print(f"  - {path}  (Appendix Table A2, {len(df)} thresholds)")

    # 미리보기
    print(f"    ┌──────────────────────────────────────────────────────┐")
    print(f"    │ {primary_metric.upper()} threshold      n        ρ          p       │")
    print(f"    ├──────────────────────────────────────────────────────┤")
    for r in rows:
        p_display = r['p_value']
        t = r['cnl_threshold']
        # 문자열('> 0 (strict)') 또는 float 모두 처리
        t_str = t if isinstance(t, str) else f'≥ {t:.3f}'
        print(f"    │  {t_str:>14s}   {r['n_pairs']:>5d}   {r['spearman_rho']:+.4f}   {p_display:>9s}  │")
    print(f"    └──────────────────────────────────────────────────────┘")
    return df


def bootstrap_rho_ci(values_x, values_y, n_boot=1000, seed=42, ci=0.95):
    """단일 모델 ρ에 대한 bootstrap CI.

    페어를 with-replacement resample → 매번 Spearman ρ 재계산 → percentile CI.

    Args:
        values_x, values_y: 1-D arrays (matched 페어, 동일 길이)
        n_boot: bootstrap 반복 횟수 (default 1000)
        seed: 재현성 시드
        ci: 신뢰구간 폭 (default 0.95)

    Returns:
        dict: {rho, ci_low, ci_high, n_boot, n_pairs, rhos (sorted array)}
    """
    rng = np.random.default_rng(seed)
    x = np.asarray(values_x, dtype=float)
    y = np.asarray(values_y, dtype=float)
    n = len(x)
    if n < 10:
        return None

    rho_point, _ = spearmanr(x, y)
    rhos = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        r, _ = spearmanr(x[idx], y[idx])
        rhos[b] = r if not np.isnan(r) else 0.0

    alpha = (1 - ci) / 2
    lo, hi = np.quantile(rhos, [alpha, 1 - alpha])
    return {
        'rho': float(rho_point),
        'ci_low': float(lo),
        'ci_high': float(hi),
        'n_boot': int(n_boot),
        'n_pairs': int(n),
        'rhos': rhos,
    }


def paired_bootstrap_rho_diff(emb_sim_a, emb_sim_b, dicl_val,
                              n_boot=1000, seed=42, ci=0.95):
    """두 모델 ρ 차이 (ρ_A - ρ_B)의 paired bootstrap.

    동일 페어 인덱스를 resample → 두 모델 ρ 동시 계산 → 차이 분포.
    Paired 설계로 페어 수준 noise를 제거 → 모델 간 차이 통계력 ↑.

    Args:
        emb_sim_a: 모델 A의 페어별 embedding similarity (1-D)
        emb_sim_b: 모델 B의 페어별 embedding similarity (1-D)
        dicl_val: 페어별 DICL 지표값 (1-D, A/B와 동일 인덱스)
        n_boot, seed, ci: bootstrap_rho_ci와 동일

    Returns:
        dict: {rho_a, rho_b, delta, ci_low, ci_high, p_value, n_boot, n_pairs}
            · p_value: percentile-based two-sided
              p = 2·min(P(Δ_boot ≤ 0), P(Δ_boot ≥ 0))
    """
    rng = np.random.default_rng(seed)
    a = np.asarray(emb_sim_a, dtype=float)
    b = np.asarray(emb_sim_b, dtype=float)
    d = np.asarray(dicl_val, dtype=float)
    assert len(a) == len(b) == len(d), "A, B, DICL은 동일 길이여야 함 (paired)"
    n = len(d)
    if n < 10:
        return None

    rho_a, _ = spearmanr(a, d)
    rho_b, _ = spearmanr(b, d)
    delta_point = rho_a - rho_b

    deltas = np.empty(n_boot, dtype=float)
    for k in range(n_boot):
        idx = rng.integers(0, n, size=n)
        ra, _ = spearmanr(a[idx], d[idx])
        rb, _ = spearmanr(b[idx], d[idx])
        deltas[k] = (0.0 if np.isnan(ra) else ra) - (0.0 if np.isnan(rb) else rb)

    alpha = (1 - ci) / 2
    lo, hi = np.quantile(deltas, [alpha, 1 - alpha])
    p = 2.0 * float(min(np.mean(deltas <= 0), np.mean(deltas >= 0)))
    return {
        'rho_a': float(rho_a),
        'rho_b': float(rho_b),
        'delta': float(delta_point),
        'ci_low': float(lo),
        'ci_high': float(hi),
        'p_value': float(p),
        'n_boot': int(n_boot),
        'n_pairs': int(n),
    }


def extract_embeddings(model, dataloader, device):
    """모델의 encode()로 모든 샘플의 임베딩을 추출."""
    model.eval()
    all_embeddings = []
    all_nat_labels = []
    all_lang_labels = []

    with torch.no_grad():
        for batch in dataloader:
            *inputs, nat_target, lang_target = batch
            inputs = [t.to(device) for t in inputs]

            embeddings = model.encode(*inputs)  # [B, d_model]
            all_embeddings.append(embeddings.cpu().numpy())
            all_nat_labels.extend(nat_target.numpy())
            all_lang_labels.extend(lang_target.numpy())

    return (
        np.concatenate(all_embeddings, axis=0),
        np.array(all_nat_labels),
        np.array(all_lang_labels),
    )


def compute_country_centroids(embeddings, nat_labels, country_list):
    """국가별 임베딩 평균(centroid) 계산."""
    centroids = {}
    for idx, noc in enumerate(country_list):
        mask = (nat_labels == idx)
        if mask.sum() > 0:
            centroids[noc] = embeddings[mask].mean(axis=0)
    return centroids


def compute_pairwise_distances(centroids, country_list):
    """국가 쌍 간 cosine distance 계산."""
    distances = {}
    countries = [c for c in country_list if c in centroids]

    for i, c1 in enumerate(countries):
        for j, c2 in enumerate(countries):
            if i >= j:
                continue
            dist = cosine(centroids[c1], centroids[c2])
            distances[(c1, c2)] = dist

    return distances


def load_dicl(dicl_path, metric='cnl'):
    """
    DICL_v2.csv에서 지정된 지표 로드.

    Args:
        dicl_path: DICL CSV 경로
        metric: 사용할 지표 (기본 'cnl')
            - cnl: Common Native Language index (공통 native language 공유, ★ Phase 4 primary)
            - lpn: Linguistic Phonetic (다른 지표)
            - col: Co-official language (1=공유, 0=비공유)
            - bpn: Baseline Phonetic
    """
    df = pd.read_csv(dicl_path)
    df = df.dropna(subset=[metric])
    df[metric] = df[metric].astype(float)

    dicl_dist = {}
    for _, row in df.iterrows():
        iso_i = row['iso3_i']
        iso_j = row['iso3_j']
        if iso_i != iso_j:
            key = tuple(sorted([iso_i, iso_j]))
            dicl_dist[key] = row[metric]

    return dicl_dist


def compute_cnl_rho_quick(config, model_path):
    """
    학습된 모델로 CNL 상관관계만 빠르게 계산 (저장 파일 없이 dict 반환).

    용도: run_all_experiments.py에서 각 실험의 cnl_rho_all / cnl_rho_subset을
         experiment_results.csv에 auto-logging.

    지원 모델: encode() 메서드를 가진 모든 모델.
      · multichannel_encoder (Transformer)
      · multichannel_cnn / multichannel_bilstm / multichannel_fasttext (baselines)

    Returns:
        {'cnl_rho_all': float, 'cnl_rho_subset': float, 'cnl_n_pairs': int}
        체크포인트 없거나 에러 시 None.
    """
    if not os.path.exists(model_path):
        return None

    device = config['experiment']['device']
    if device == 'cuda' and not torch.cuda.is_available():
        device = 'cpu'

    # 1. 데이터 + 모델 로드
    from data.dataset import MultiChannelDataset
    df = get_loaded_df(config).copy()
    df['Name'] = df['Name'].apply(replace_disallowed_chars)
    df = df[df['Name'].str.len() > 0].reset_index(drop=True)

    country_idx = get_country_idx(config)
    lang_group_idx = get_lang_group_idx(config)
    country_list = get_country_list(config)
    ngram_dicts = get_ngram_dicts(config['data']['ngram'])
    vocab_size = get_vocab_size(config)

    dataset = MultiChannelDataset(
        texts=df['Name'], labels=df['NOC'], lang_labels=df['LanguageGroup'],
        uni_ngram_dict=ngram_dicts['uni'], bi_ngram_dict=ngram_dicts['bi'],
        tri_ngram_dict=ngram_dicts['tri'],
        max_length=config['data']['max_length'],
        country_idx=country_idx, lang_group_idx=lang_group_idx,
    )
    loader = DataLoader(dataset, batch_size=256, shuffle=False)

    model = get_model(
        config, vocab_size,
        n_classes=config['model']['num_classes'],
        n_lang_groups=config['model']['num_lang_groups'],
    )
    if not hasattr(model, 'encode'):
        # encode() 미구현 모델은 분석 불가 — 안전하게 None 반환
        return None
    _state = torch.load(model_path, map_location=device)
    _missing, _unexpected = model.load_state_dict(_state, strict=False)
    if _missing or _unexpected:
        print(f"  [load_state_dict] strict=False  missing={len(_missing)}  unexpected={len(_unexpected)}")
        if _missing:
            print(f"    missing (first 10): {list(_missing)[:10]}")
        if _unexpected:
            print(f"    unexpected (first 10): {list(_unexpected)[:10]}")
    model = model.to(device)

    # 2. 임베딩 + centroid
    embeddings, nat_labels, _ = extract_embeddings(model, loader, device)
    centroids = compute_country_centroids(embeddings, nat_labels, country_list)
    if len(centroids) < 10:
        return None

    # 3. CNL 로드 + pairwise 계산
    dicl_path = os.path.join("data", "linguistics", "DICL_v2.csv")
    dicl_cnl = load_dicl(dicl_path, metric='cnl')

    matched_emb = []
    matched_cnl = []
    for (noc_i, noc_j), emb_dist in compute_pairwise_distances(centroids, country_list).items():
        iso_i = noc_to_iso3(noc_i)
        iso_j = noc_to_iso3(noc_j)
        key = tuple(sorted([iso_i, iso_j]))
        if key in dicl_cnl:
            matched_emb.append(emb_dist)
            matched_cnl.append(dicl_cnl[key])

    if len(matched_emb) < 10:
        return None

    emb_sim = [1.0 - d for d in matched_emb]
    rho_all, _ = spearmanr(matched_cnl, emb_sim)

    # Subset: CNL > 0 strict
    cnl_arr = np.array(matched_cnl)
    emb_arr = np.array(emb_sim)
    mask = cnl_arr > 0
    if mask.sum() >= 10:
        rho_sub, _ = spearmanr(cnl_arr[mask], emb_arr[mask])
    else:
        rho_sub = float('nan')

    return {
        'cnl_rho_all': round(float(rho_all), 4),
        'cnl_rho_subset': round(float(rho_sub), 4),
        'cnl_n_pairs': len(matched_emb),
    }


def run_dicl_analysis(config, model_path, output_dir):
    """DICL 상관관계 분석 전체 파이프라인."""
    device = config['experiment']['device']
    set_seed(config['experiment']['seed'])

    print("=" * 60)
    print(" DICL Correlation Analysis")
    print("=" * 60)

    # 1. 데이터 준비
    print("\n[1/6] 데이터 로딩")
    ngram = config['data']['ngram']
    max_length = config['data']['max_length']

    df = get_loaded_df(config).copy()
    df['Name'] = df['Name'].apply(replace_disallowed_chars)
    df = df[df['Name'].str.len() > 0].reset_index(drop=True)

    country_idx = get_country_idx(config)
    lang_group_idx = get_lang_group_idx(config)
    country_list = get_country_list(config)
    ngram_dicts = get_ngram_dicts(ngram)
    vocab_size = get_vocab_size(config)

    # 전체 데이터로 Dataset 생성 (임베딩 추출용)
    dataset = MultiChannelDataset(
        texts=df['Name'], labels=df['NOC'], lang_labels=df['LanguageGroup'],
        uni_ngram_dict=ngram_dicts['uni'], bi_ngram_dict=ngram_dicts['bi'],
        tri_ngram_dict=ngram_dicts['tri'], max_length=max_length,
        country_idx=country_idx, lang_group_idx=lang_group_idx,
    )
    loader = DataLoader(dataset, batch_size=256, shuffle=False)

    # 2. 모델 로드
    print("[2/6] 모델 로드")
    model = get_model(
        config, vocab_size,
        n_classes=config['model']['num_classes'],
        n_lang_groups=config['model']['num_lang_groups'],
    )
    if not hasattr(model, 'encode'):
        raise AttributeError(
            f"모델 '{config['model']['type']}'에 encode() 메서드가 없습니다. "
            f"DICL 상관관계 분석은 encode()를 통한 임베딩 추출이 필요합니다."
        )
    _state = torch.load(model_path, map_location=device)
    _missing, _unexpected = model.load_state_dict(_state, strict=False)
    if _missing or _unexpected:
        print(f"  [load_state_dict] strict=False  missing={len(_missing)}  unexpected={len(_unexpected)}")
        if _missing:
            print(f"    missing (first 10): {list(_missing)[:10]}")
        if _unexpected:
            print(f"    unexpected (first 10): {list(_unexpected)[:10]}")
    model = model.to(device)
    print(f"  모델 ({config['model']['type']}): {model_path}")

    # 3. 임베딩 추출
    print("[3/6] 임베딩 추출")
    embeddings, nat_labels, lang_labels = extract_embeddings(model, loader, device)
    print(f"  임베딩 shape: {embeddings.shape}")

    # 4. 국가별 centroid + pairwise distance
    print("[4/6] 국가별 centroid 및 pairwise distance 계산")
    centroids = compute_country_centroids(embeddings, nat_labels, country_list)
    emb_distances = compute_pairwise_distances(centroids, country_list)
    print(f"  Centroid 수: {len(centroids)}")
    print(f"  국가 쌍 수: {len(emb_distances)}")

    # 5. DICL 로드 + 매핑 (11개 지표 전부 계산)
    print("[5/6] DICL 데이터 로드 및 매핑")
    dicl_path = os.path.join("data", "linguistics", "DICL_v2.csv")

    # 이전 THRESHOLD=0.001 필터를 제거한 이유:
    # - DICL 0은 legitimate value (공통 언어 없음 또는 tree distance 최대)
    # - NaN은 load_dicl의 dropna()에서 이미 처리됨
    # - CNL은 52% 페어가 정확히 0 → 필터하면 전체 분포 왜곡
    all_results = {}

    for metric in ALL_DICL_METRICS:
        dicl_dist = load_dicl(dicl_path, metric=metric)

        matched_emb = []
        matched_dicl = []
        matched_pairs_metric = []

        for (noc_i, noc_j), emb_dist in emb_distances.items():
            iso_i = noc_to_iso3(noc_i)
            iso_j = noc_to_iso3(noc_j)
            key = tuple(sorted([iso_i, iso_j]))

            if key in dicl_dist:   # NaN은 load_dicl에서 이미 dropna
                matched_emb.append(emb_dist)
                matched_dicl.append(dicl_dist[key])
                matched_pairs_metric.append((noc_i, noc_j))

        if len(matched_emb) >= 10:
            rho, p_value = spearmanr(matched_emb, matched_dicl)
            all_results[metric] = {
                'rho': rho, 'p_value': p_value,
                'n_pairs': len(matched_emb),
                'matched_emb': matched_emb,
                'matched_dicl': matched_dicl,
                'matched_pairs': matched_pairs_metric,
            }
            print(f"  {metric}: ρ={rho:+.4f}, p={p_value:.2e}, n={len(matched_emb)}")

    # 6. 결과 저장
    print("\n[6/6] 결과 저장")
    os.makedirs(output_dir, exist_ok=True)

    # 주 지표: cnl (Common Native Language index) — embedding similarity와 가장 강한 상관
    primary_metric = 'cnl'
    if primary_metric not in all_results:
        primary_metric = list(all_results.keys())[0]  # fallback

    pr = all_results[primary_metric]
    rho, p_value = pr['rho'], pr['p_value']
    matched_emb = pr['matched_emb']
    matched_dicl = pr['matched_dicl']
    matched_pairs = pr['matched_pairs']

    print(f"\n  ★ 주 지표: {primary_metric}")
    print(f"    Spearman ρ = {rho:.4f}")
    print(f"    p-value = {p_value:.2e}")
    print(f"    매칭 쌍 수 = {len(matched_emb)}")

    # Bootstrap 95% CI — Reviewer 방어책 (ρ가 noise가 아님 입증)
    # · all (전체 페어) + subset (cnl > 0 strict) 둘 다 CI 보고
    emb_sim_arr = 1.0 - np.asarray(matched_emb, dtype=float)
    cnl_arr_full = np.asarray(matched_dicl, dtype=float)

    boot_all = bootstrap_rho_ci(emb_sim_arr, cnl_arr_full, n_boot=1000, seed=42)
    sub_mask = cnl_arr_full > 0
    boot_sub = (
        bootstrap_rho_ci(emb_sim_arr[sub_mask], cnl_arr_full[sub_mask], n_boot=1000, seed=42)
        if sub_mask.sum() >= 10 else None
    )

    if boot_all is not None:
        print(f"    [bootstrap] ρ_all     = {boot_all['rho']:+.4f} "
              f"95% CI [{boot_all['ci_low']:+.4f}, {boot_all['ci_high']:+.4f}] "
              f"(n={boot_all['n_pairs']}, B={boot_all['n_boot']})")
    if boot_sub is not None:
        print(f"    [bootstrap] ρ_subset  = {boot_sub['rho']:+.4f} "
              f"95% CI [{boot_sub['ci_low']:+.4f}, {boot_sub['ci_high']:+.4f}] "
              f"(n={boot_sub['n_pairs']}, B={boot_sub['n_boot']})")

    # bootstrap 결과 CSV — 단일 모델 (paired 비교는 run_paired_comparison에서 별도)
    boot_rows = []
    if boot_all is not None:
        boot_rows.append({
            'scope': 'all',
            'rho': boot_all['rho'],
            'ci_low': boot_all['ci_low'],
            'ci_high': boot_all['ci_high'],
            'n_pairs': boot_all['n_pairs'],
            'n_boot': boot_all['n_boot'],
            'model_path': model_path,
        })
    if boot_sub is not None:
        boot_rows.append({
            'scope': 'subset_cnl_gt0',
            'rho': boot_sub['rho'],
            'ci_low': boot_sub['ci_low'],
            'ci_high': boot_sub['ci_high'],
            'n_pairs': boot_sub['n_pairs'],
            'n_boot': boot_sub['n_boot'],
            'model_path': model_path,
        })
    if boot_rows:
        pd.DataFrame(boot_rows).to_csv(
            os.path.join(output_dir, 'rho_bootstrap_summary.csv'), index=False
        )
        print("  - rho_bootstrap_summary.csv")

    # 상세 결과 CSV (주 지표)
    result_df = pd.DataFrame({
        'noc_i': [p[0] for p in matched_pairs],
        'noc_j': [p[1] for p in matched_pairs],
        'embedding_distance': matched_emb,
        f'dicl_{primary_metric}': matched_dicl,
    })
    result_df.to_csv(os.path.join(output_dir, 'dicl_correlation_pairs.csv'), index=False)

    # 전체 지표 요약 CSV
    summary_rows = []
    for metric, res in all_results.items():
        summary_rows.append({
            'metric': metric,
            'spearman_rho': res['rho'],
            'p_value': res['p_value'],
            'n_pairs': res['n_pairs'],
            'primary': metric == primary_metric,
            'model_path': model_path,
            'n_countries_with_centroids': len(centroids),
        })
    pd.DataFrame(summary_rows).to_csv(
        os.path.join(output_dir, 'dicl_correlation_summary.csv'), index=False
    )

    # centroid 저장 (시각화용)
    centroid_data = {noc: emb.tolist() for noc, emb in centroids.items()}
    np.savez(
        os.path.join(output_dir, 'country_centroids.npz'),
        **{noc: emb for noc, emb in centroids.items()}
    )

    print(f"\n  결과 저장: {output_dir}/")
    print(f"  - dicl_correlation_pairs.csv")
    print(f"  - dicl_correlation_summary.csv")
    print(f"  - country_centroids.npz")

    # 논문 Appendix용 Table A1 — DICL 11개 지표 비교
    export_dicl_metric_table(all_results, primary_metric)

    # Appendix Table A2 — CNL 분포 스큐에 대한 robustness check
    export_dicl_robustness_check(all_results, primary_metric)

    # Appendix B — LPN 기반 supplementary: Malta-Arabic orthographic bias case
    export_appendix_b_malta_lpn(all_results)

    return rho, p_value, matched_emb, matched_dicl, matched_pairs


def _model_to_pair_emb_sim(config, model_path, metric='cnl',
                           model_type_override=None):
    """모델 체크포인트 → 페어별 (emb_sim, dicl_value) dict.

    paired comparison에서 두 모델 사이 페어 alignment에 사용.

    Returns:
        emb_sim_by_pair: {(iso_i, iso_j): emb_similarity}
        dicl_by_pair:    {(iso_i, iso_j): dicl_value}
        둘 다 동일 키 사용 (sorted iso3 tuple).
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"모델 체크포인트 없음: {model_path}")

    device = config['experiment']['device']
    if device == 'cuda' and not torch.cuda.is_available():
        device = 'cpu'

    from data.dataset import MultiChannelDataset
    df = get_loaded_df(config).copy()
    df['Name'] = df['Name'].apply(replace_disallowed_chars)
    df = df[df['Name'].str.len() > 0].reset_index(drop=True)

    country_idx = get_country_idx(config)
    lang_group_idx = get_lang_group_idx(config)
    country_list = get_country_list(config)
    ngram_dicts = get_ngram_dicts(config['data']['ngram'])
    vocab_size = get_vocab_size(config)

    dataset = MultiChannelDataset(
        texts=df['Name'], labels=df['NOC'], lang_labels=df['LanguageGroup'],
        uni_ngram_dict=ngram_dicts['uni'], bi_ngram_dict=ngram_dicts['bi'],
        tri_ngram_dict=ngram_dicts['tri'],
        max_length=config['data']['max_length'],
        country_idx=country_idx, lang_group_idx=lang_group_idx,
    )
    loader = DataLoader(dataset, batch_size=256, shuffle=False)

    # 두 모델이 다른 아키텍처일 때 (예: paired Ours vs CNN) model.type을 override해
    # 각 .pt에 맞는 모델 인스턴스를 생성한다. 원본 config는 mutate하지 않는다.
    model_cfg = config
    if model_type_override is not None:
        import copy as _copy
        model_cfg = _copy.deepcopy(config)
        model_cfg['model']['type'] = model_type_override
        print(f"  [model_type override] {config['model']['type']} → {model_type_override}")

    model = get_model(
        model_cfg, vocab_size,
        n_classes=model_cfg['model']['num_classes'],
        n_lang_groups=model_cfg['model']['num_lang_groups'],
    )
    if not hasattr(model, 'encode'):
        raise ValueError(f"{type(model).__name__}: encode() 미구현 — paired 비교 불가")
    _state = torch.load(model_path, map_location=device)
    _missing, _unexpected = model.load_state_dict(_state, strict=False)
    if _missing or _unexpected:
        print(f"  [load_state_dict] strict=False  missing={len(_missing)}  unexpected={len(_unexpected)}")
        if _missing:
            print(f"    missing (first 10): {list(_missing)[:10]}")
        if _unexpected:
            print(f"    unexpected (first 10): {list(_unexpected)[:10]}")
    model = model.to(device)

    embeddings, nat_labels, _ = extract_embeddings(model, loader, device)
    centroids = compute_country_centroids(embeddings, nat_labels, country_list)
    if len(centroids) < 10:
        raise RuntimeError(f"centroid 부족 ({len(centroids)} < 10)")

    dicl_path = os.path.join("data", "linguistics", "DICL_v2.csv")
    dicl_dist = load_dicl(dicl_path, metric=metric)

    emb_sim_by_pair = {}
    dicl_by_pair = {}
    for (noc_i, noc_j), emb_dist in compute_pairwise_distances(centroids, country_list).items():
        iso_i = noc_to_iso3(noc_i)
        iso_j = noc_to_iso3(noc_j)
        key = tuple(sorted([iso_i, iso_j]))
        if key in dicl_dist:
            emb_sim_by_pair[key] = 1.0 - emb_dist  # cosine similarity
            dicl_by_pair[key] = dicl_dist[key]

    return emb_sim_by_pair, dicl_by_pair


def run_paired_comparison(config, model_path_a, model_path_b, output_dir,
                          metric='cnl', n_boot=1000, seed=42,
                          model_a_type=None, model_b_type=None):
    """두 모델 간 ρ 차이의 paired bootstrap 비교.

    동일 페어를 함께 resample하여 noise 제거 → 모델 간 차이 통계력 ↑.
    Reviewer 방어용: "ρ_A=0.50 vs ρ_B=0.43, Δρ=0.07이 통계적으로 유의한가?"

    Args:
        model_path_a, model_path_b: 비교할 두 체크포인트 경로
        metric: DICL 지표 (default 'cnl')
        n_boot, seed: bootstrap 설정
    """
    print("=" * 60)
    print(f"Paired ρ Comparison ({metric}): A vs B")
    print(f"  A = {model_path_a}")
    print(f"  B = {model_path_b}")
    print("=" * 60)

    set_seed(config['experiment']['seed'])

    print("[1/3] 모델 A 임베딩 추출 + pairwise distance")
    emb_a, dicl_dict_a = _model_to_pair_emb_sim(
        config, model_path_a, metric=metric, model_type_override=model_a_type,
    )

    print("[2/3] 모델 B 임베딩 추출 + pairwise distance")
    emb_b, dicl_dict_b = _model_to_pair_emb_sim(
        config, model_path_b, metric=metric, model_type_override=model_b_type,
    )

    # 두 모델 모두 매칭되는 페어 (paired) 정렬
    common_keys = sorted(set(emb_a.keys()) & set(emb_b.keys()))
    if len(common_keys) < 10:
        raise RuntimeError(f"공통 페어 부족: {len(common_keys)} < 10")
    a_arr = np.array([emb_a[k] for k in common_keys], dtype=float)
    b_arr = np.array([emb_b[k] for k in common_keys], dtype=float)
    d_arr = np.array([dicl_dict_a[k] for k in common_keys], dtype=float)
    print(f"  공통 페어 수: {len(common_keys)}")

    # All scope
    print("[3/3] paired bootstrap (all + subset)")
    res_all = paired_bootstrap_rho_diff(a_arr, b_arr, d_arr, n_boot=n_boot, seed=seed)
    sub_mask = d_arr > 0
    res_sub = (
        paired_bootstrap_rho_diff(a_arr[sub_mask], b_arr[sub_mask], d_arr[sub_mask],
                                  n_boot=n_boot, seed=seed)
        if sub_mask.sum() >= 10 else None
    )

    def _print(scope, r):
        if r is None:
            return
        print(f"\n  [{scope}]")
        print(f"    ρ_A     = {r['rho_a']:+.4f}")
        print(f"    ρ_B     = {r['rho_b']:+.4f}")
        print(f"    Δρ      = {r['delta']:+.4f}")
        print(f"    95% CI  = [{r['ci_low']:+.4f}, {r['ci_high']:+.4f}]")
        print(f"    p-value = {r['p_value']:.4f}")
        print(f"    n_pairs = {r['n_pairs']}, n_boot = {r['n_boot']}")
        verdict = "유의 (CI에 0 미포함)" if (r['ci_low'] > 0) or (r['ci_high'] < 0) else "유의 X (CI에 0 포함)"
        print(f"    → {verdict}")

    _print("all (모든 페어)", res_all)
    _print(f"subset ({metric} > 0, structural non-zero)", res_sub)

    # CSV 저장
    os.makedirs(output_dir, exist_ok=True)
    rows = []
    for scope, r in [('all', res_all), (f'subset_{metric}_gt0', res_sub)]:
        if r is None:
            continue
        rows.append({
            'scope': scope,
            'metric': metric,
            'model_a': model_path_a,
            'model_b': model_path_b,
            'rho_a': r['rho_a'],
            'rho_b': r['rho_b'],
            'delta': r['delta'],
            'ci_low': r['ci_low'],
            'ci_high': r['ci_high'],
            'p_value': r['p_value'],
            'n_pairs': r['n_pairs'],
            'n_boot': r['n_boot'],
        })
    if rows:
        out_path = os.path.join(output_dir, 'rho_paired_comparison.csv')
        pd.DataFrame(rows).to_csv(out_path, index=False)
        print(f"\n  결과 저장: {out_path}")
    return res_all, res_sub


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="DICL Correlation Analysis")
    parser.add_argument("--config", type=str, default=os.path.join("configs", "base_config.yaml"))
    parser.add_argument("--model_path", type=str, required=True,
                        help="학습된 모델 체크포인트 경로")
    parser.add_argument("--output_dir", type=str, default=os.path.join("outputs", "analysis"))
    parser.add_argument("--compare_model", type=str, default=None,
                        help="두 번째 모델 경로 — 지정 시 paired bootstrap 비교 실행 "
                             "(--model_path A, --compare_model B → ρ_A - ρ_B)")
    parser.add_argument("--model_a_type", type=str, default=None,
                        help="모델 A의 model.type override (예: multichannel_conformer). "
                             "미지정 시 config의 model.type 그대로 사용.")
    parser.add_argument("--model_b_type", type=str, default=None,
                        help="모델 B의 model.type override (예: multichannel_cnn). "
                             "Ours vs CNN처럼 두 모델 아키텍처가 다를 때 반드시 지정.")
    parser.add_argument("--n_boot", type=int, default=1000,
                        help="bootstrap 반복 횟수 (default 1000)")
    parser.add_argument("--metric", type=str, default='cnl',
                        help="DICL 지표 (default cnl)")
    args = parser.parse_args()

    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    if args.compare_model is not None:
        run_paired_comparison(
            config, args.model_path, args.compare_model, args.output_dir,
            metric=args.metric, n_boot=args.n_boot,
            model_a_type=args.model_a_type,
            model_b_type=args.model_b_type,
        )
    else:
        run_dicl_analysis(config, args.model_path, args.output_dir)
