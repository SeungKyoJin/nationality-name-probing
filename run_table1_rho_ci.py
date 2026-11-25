"""
Table 1 전 모델 ρ bootstrap CI 일괄 계산 (rebuttal R1, Reviewer xhzC/mDJP 대응).

각 Table 1 행의 논문 수치를 만든 seed-42 checkpoint에서:
  1. 전체 데이터 임베딩 → NOC centroid → pairwise cosine → DICL cnl 매칭
  2. pair-resampling bootstrap 95% CI (ρ_all / ρ_subset, 1,000회)
  3. Ours vs 각 baseline paired bootstrap Δρ (동일 페어)

seed 축(mean±std, Appendix T8)과 estimator 축(bootstrap CI)은 별개 —
여기서는 후자만 다룬다. CI는 geometry 분석(Fig 2)과 동일한 seed-42
checkpoint 기준이며, 요약 CSV에 checkpoint 절대경로를 기록해 출처를 남긴다.

실행:
  python run_table1_rho_ci.py            # 6개 신경망 모델 전부
  python run_table1_rho_ci.py --n_boot 1000
"""

import argparse
import copy
import os
import time

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

from analysis.dicl_correlation import (
    bootstrap_rho_ci, paired_bootstrap_rho_diff,
    extract_embeddings, compute_country_centroids,
    compute_pairwise_distances, load_dicl,
)
from analysis.noc_iso3_mapping import noc_to_iso3
from data.data_loader import (
    get_loaded_df, get_country_idx, get_country_list, get_lang_group_idx,
    get_ngram_dicts, get_vocab_size,
)
from data.dataset import MultiChannelDataset
from models import get_model
from utils.util import replace_disallowed_chars, set_seed

# 논문 최종 checkpoint 보관 위치 (원본 실험 레포 — 읽기 전용)
CKPT_ROOT = "c:/Users/t4716/PycharmProjects/nationality_classifier/outputs/checkpoints"
OUTPUT_DIR = os.path.join("outputs", "analysis", "table1_rho_ci")
DICL_PATH = os.path.join("data", "linguistics", "DICL_v2.csv")

# Table 1 행 ↔ checkpoint ↔ 모델 config 오버라이드.
# 아키텍처 차원(embedding 256 / hidden 1024 / head 8 / layer 4)은 전 모델 공통
# (base_config 값 그대로). strict=True 로드가 실패하면 config 불일치이므로 즉시 중단.
TABLE1_MODELS = [
    # (label, checkpoint 상대경로, model 오버라이드)
    ("fasttext", "260513/baseline_fasttext.pt",
     {"type": "multichannel_fasttext", "fusion_type": "linear", "pool_type": "attention"}),
    ("bilstm", "260513/baseline_bilstm.pt",
     {"type": "multichannel_bilstm", "fusion_type": "linear", "pool_type": "attention"}),
    ("cnn", "260513/baseline_cnn.pt",
     {"type": "multichannel_cnn", "fusion_type": "linear", "pool_type": "attention"}),
    ("transformer", "260513/ours_transformer.pt",
     {"type": "multichannel_encoder", "fusion_type": "linear", "pool_type": "attention"}),
    ("vanilla_mcc", "260516/conf_ensemble_off.pt",
     {"type": "multichannel_conformer", "channel_ensemble_head": False}),
    ("ours", "260516/multiseed_ours_alpha03_seed42.pt",
     {"type": "multichannel_conformer", "channel_ensemble_head": True,
      "channel_fusion_type": "uniform"}),
]


def build_full_loader(config, batch_size=256):
    """전체 데이터(243K)에 대한 MultiChannelDataset loader — 전 모델 공유."""
    df = get_loaded_df(config).copy()
    df["Name"] = df["Name"].apply(replace_disallowed_chars)
    df = df[df["Name"].str.len() > 0].reset_index(drop=True)

    ngram_dicts = get_ngram_dicts("multi")
    dataset = MultiChannelDataset(
        texts=df["Name"], labels=df["NOC"], lang_labels=df["LanguageGroup"],
        uni_ngram_dict=ngram_dicts["uni"], bi_ngram_dict=ngram_dicts["bi"],
        tri_ngram_dict=ngram_dicts["tri"],
        max_length=config["data"]["max_length"],
        country_idx=get_country_idx(config),
        lang_group_idx=get_lang_group_idx(config),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=False), df


def load_model(base_config, model_overrides, ckpt_path, device):
    """오버라이드 적용 config로 모델 생성 후 strict=True 로드."""
    config = copy.deepcopy(base_config)
    config["model"].update(model_overrides)

    model = get_model(
        config, get_vocab_size(config),
        n_classes=config["model"]["num_classes"],
        n_lang_groups=config["model"]["num_lang_groups"],
    )
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state, strict=True)   # 불일치 시 즉시 실패 → config 오류 검출
    return model.to(device).eval()


def compute_pairs(model, loader, df, country_list, dicl_cnl, device):
    """임베딩 → centroid → DICL 매칭 페어 테이블 (noc_i, noc_j, emb_sim, cnl)."""
    embeddings, nat_labels, _ = extract_embeddings(model, loader, device)
    centroids = compute_country_centroids(embeddings, nat_labels, country_list)

    rows = []
    for (noc_i, noc_j), emb_dist in compute_pairwise_distances(centroids, country_list).items():
        key = tuple(sorted([noc_to_iso3(noc_i), noc_to_iso3(noc_j)]))
        if key in dicl_cnl:
            rows.append({"noc_i": noc_i, "noc_j": noc_j,
                         "emb_sim": 1.0 - emb_dist, "cnl": dicl_cnl[key]})
    return pd.DataFrame(rows)


def ci_rows_for(label, ckpt_path, pairs, n_boot):
    """all / subset 두 scope의 bootstrap CI 요약 행 생성."""
    rows = []
    scopes = [("all", pairs), ("subset_cnl_gt0", pairs[pairs["cnl"] > 0])]
    for scope, sub in scopes:
        res = bootstrap_rho_ci(sub["cnl"].values, sub["emb_sim"].values, n_boot=n_boot)
        rows.append({
            "model": label, "scope": scope,
            "rho": round(res["rho"], 4),
            "ci_low": round(res["ci_low"], 4),
            "ci_high": round(res["ci_high"], 4),
            "n_pairs": res["n_pairs"], "n_boot": n_boot,
            "checkpoint": ckpt_path,
        })
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=os.path.join("configs", "base_config.yaml"))
    parser.add_argument("--n_boot", type=int, default=1000)
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        base_config = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    set_seed(base_config["experiment"]["seed"])
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    loader, df = build_full_loader(base_config)
    country_list = get_country_list(base_config)
    dicl_cnl = load_dicl(DICL_PATH, metric="cnl")
    print(f"[Data] {len(df)} names, {len(country_list)} NOCs")

    summary_rows, model_pairs = [], {}
    for label, rel_path, overrides in TABLE1_MODELS:
        ckpt_path = os.path.join(CKPT_ROOT, rel_path)
        if not os.path.exists(ckpt_path):
            print(f"[SKIP] {label}: checkpoint 없음 → {ckpt_path}")
            continue

        t0 = time.time()
        model = load_model(base_config, overrides, ckpt_path, device)
        pairs = compute_pairs(model, loader, df, country_list, dicl_cnl, device)
        del model
        torch.cuda.empty_cache()

        pairs.to_csv(os.path.join(OUTPUT_DIR, f"pairs_{label}.csv"), index=False)
        model_pairs[label] = pairs
        summary_rows.extend(ci_rows_for(label, ckpt_path, pairs, args.n_boot))
        done = summary_rows[-2:]
        print(f"[{label}] rho_all={done[0]['rho']} CI[{done[0]['ci_low']}, {done[0]['ci_high']}]  "
              f"rho_sub={done[1]['rho']} CI[{done[1]['ci_low']}, {done[1]['ci_high']}]  "
              f"({time.time() - t0:.0f}s)")

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(os.path.join(OUTPUT_DIR, "rho_ci_summary.csv"), index=False)

    # ─── Ours vs 각 baseline paired bootstrap (동일 페어 정렬 후) ───
    paired_rows = []
    if "ours" in model_pairs:
        ours = model_pairs["ours"].rename(columns={"emb_sim": "emb_sim_ours"})
        for label, pairs in model_pairs.items():
            if label == "ours":
                continue
            merged = ours.merge(pairs, on=["noc_i", "noc_j", "cnl"], how="inner")
            for scope, sub in [("all", merged), ("subset_cnl_gt0", merged[merged["cnl"] > 0])]:
                res = paired_bootstrap_rho_diff(
                    sub["emb_sim_ours"].values, sub["emb_sim"].values, sub["cnl"].values,
                    n_boot=args.n_boot,
                )
                paired_rows.append({
                    "scope": scope, "baseline": label,
                    "rho_ours": round(res["rho_a"], 4), "rho_baseline": round(res["rho_b"], 4),
                    "delta": round(res["delta"], 4),
                    "ci_low": round(res["ci_low"], 4), "ci_high": round(res["ci_high"], 4),
                    "p_value": res["p_value"], "n_pairs": res["n_pairs"],
                })
        pd.DataFrame(paired_rows).to_csv(
            os.path.join(OUTPUT_DIR, "rho_paired_vs_ours.csv"), index=False)

    print("\n===== Table 1 ρ bootstrap CI 요약 =====")
    print(summary.to_string(index=False))
    if paired_rows:
        print("\n===== Paired delta-rho (Ours - baseline) =====")
        print(pd.DataFrame(paired_rows).to_string(index=False))
    print(f"\n저장: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
