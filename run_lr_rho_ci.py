"""
LR (TF-IDF) baseline의 ρ bootstrap CI — Table 1 마지막 행 (rebuttal R1).

LR은 checkpoint가 없지만 seed 고정 시 결정적이므로 재학습으로 복원한다.
재계산된 ρ가 논문 값(ρ_all 0.2598 / ρ_sub 0.3328)과 일치하는지 검증한 뒤,
neural 모델들과 동일한 pair-resampling bootstrap CI를 계산한다.

country representation: analysis/dicl_lr.py와 동일하게 LR coefficient vector
(coef_[k] = NOC k의 feature 가중치) 간 cosine similarity 사용.

실행: python run_lr_rho_ci.py
"""

import os

import numpy as np
import pandas as pd
import yaml
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity

from analysis.dicl_lr import load_train_split, load_dicl_cnl
from analysis.dicl_correlation import bootstrap_rho_ci
from analysis.noc_iso3_mapping import noc_to_iso3
from utils.util import set_seed

OUTPUT_DIR = os.path.join("outputs", "analysis", "table1_rho_ci")
CONFIG_PATH = os.path.join("configs", "base_config.yaml")
N_BOOT = 1000
# 논문 Table 1 LR 행 (master_results_39group.csv, 260516/lr_baseline)
PAPER_RHO_ALL = 0.2598
PAPER_RHO_SUB = 0.3328


def main():
    config = yaml.safe_load(open(CONFIG_PATH, encoding="utf-8"))
    set_seed(config["experiment"]["seed"])

    print("[1/3] LR 재학습 (run_baseline_lr.py와 동일 세팅)")
    train_df = load_train_split(config)
    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(1, 3),
                                 min_df=2, max_features=20000, lowercase=True)
    features = vectorizer.fit_transform(train_df["Name"])
    lr = LogisticRegression(max_iter=1000, C=1.0, solver="saga", n_jobs=-1)
    lr.fit(features, train_df["NOC"])

    print("[2/3] coefficient cosine → DICL cnl 페어 테이블")
    countries = list(lr.classes_)
    sim_matrix = cosine_similarity(lr.coef_)
    iso_to_idx = {noc_to_iso3(noc): i for i, noc in enumerate(countries) if noc_to_iso3(noc)}
    idx_to_noc = {i: noc for i, noc in enumerate(countries)}

    dicl = load_dicl_cnl()
    rows = []
    for _, row in dicl.iterrows():
        iso_i, iso_j = row["iso3_i"], row["iso3_j"]
        if iso_i in iso_to_idx and iso_j in iso_to_idx:
            idx_i, idx_j = iso_to_idx[iso_i], iso_to_idx[iso_j]
            rows.append({"noc_i": idx_to_noc[idx_i], "noc_j": idx_to_noc[idx_j],
                         "emb_sim": sim_matrix[idx_i, idx_j], "cnl": row["cnl"]})
    pairs = pd.DataFrame(rows)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    pairs.to_csv(os.path.join(OUTPUT_DIR, "pairs_lr.csv"), index=False)

    print("[3/3] 검증 + bootstrap CI")
    summary_rows = []
    for scope, sub in [("all", pairs), ("subset_cnl_gt0", pairs[pairs["cnl"] > 0])]:
        res = bootstrap_rho_ci(sub["cnl"].values, sub["emb_sim"].values, n_boot=N_BOOT)
        summary_rows.append({
            "model": "lr", "scope": scope,
            "rho": round(res["rho"], 4),
            "ci_low": round(res["ci_low"], 4), "ci_high": round(res["ci_high"], 4),
            "n_pairs": res["n_pairs"], "n_boot": N_BOOT,
            "checkpoint": "(retrained; deterministic seed 42)",
        })

    rho_all, rho_sub = summary_rows[0]["rho"], summary_rows[1]["rho"]
    match_all = abs(rho_all - PAPER_RHO_ALL) < 0.002
    match_sub = abs(rho_sub - PAPER_RHO_SUB) < 0.002
    print(f"  rho_all={rho_all} (paper {PAPER_RHO_ALL}) -> {'MATCH' if match_all else 'MISMATCH!'}")
    print(f"  rho_sub={rho_sub} (paper {PAPER_RHO_SUB}) -> {'MATCH' if match_sub else 'MISMATCH!'}")

    summary_path = os.path.join(OUTPUT_DIR, "rho_ci_summary.csv")
    existing = pd.read_csv(summary_path) if os.path.exists(summary_path) else pd.DataFrame()
    existing = existing[existing.get("model", pd.Series(dtype=str)) != "lr"]  # 재실행 시 중복 방지
    pd.concat([existing, pd.DataFrame(summary_rows)], ignore_index=True).to_csv(
        summary_path, index=False)

    print(pd.DataFrame(summary_rows).to_string(index=False))
    print(f"\n저장: {summary_path}")


if __name__ == "__main__":
    main()
