"""
α ablation multi-seed 평가 (rebuttal R2, Reviewer HUuH 대응).

HUuH: "α=0 ablation이 단일 시드로만 보고됐다" →
outputs/checkpoints/pts/의 multiseed_alpha_{α}_seed{42,7,123}.pt (학습 완료,
결과 미기록)를 평가해서 test 지표 + cnl ρ를 시드별로 산출한다. 학습 없음.

주의: multi-seed 실험은 시드가 robust_split까지 바꾸므로, 각 checkpoint는
반드시 자기 시드의 test split에서 평가한다 (analysis/confusion_asymmetry.py에서
검증된 재현 절차와 동일). cnl ρ는 전체 데이터 기준이라 split과 무관.

실행:
  python run_alpha_multiseed_eval.py                     # α=0.0만 (HUuH 대응)
  python run_alpha_multiseed_eval.py --alphas 0.0 0.5 0.7  # 전체 α ablation
"""

import argparse
import os
import time

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

from analysis.dicl_correlation import (
    bootstrap_rho_ci, extract_embeddings, compute_country_centroids,
    compute_pairwise_distances, load_dicl,
)
from analysis.noc_iso3_mapping import noc_to_iso3
from data.data_loader import (
    get_loaded_df, get_country_idx, get_country_list, get_lang_group_idx,
    get_ngram_dicts, robust_split,
)
from data.dataset import MultiChannelDataset
from run_table1_rho_ci import load_model
from utils.util import replace_disallowed_chars, set_seed

CKPT_DIR = os.path.join("outputs", "checkpoints", "pts")
OUTPUT_DIR = os.path.join("outputs", "analysis", "alpha_multiseed")
DICL_PATH = os.path.join("data", "linguistics", "DICL_v2.csv")
SEEDS = (42, 7, 123)
# α는 loss 가중치일 뿐 아키텍처는 FINAL Conformer와 동일 (strict load로 검증됨)
CONFORMER_OVERRIDES = {
    "type": "multichannel_conformer",
    "channel_ensemble_head": True,
    "channel_fusion_type": "uniform",
}


def build_dataset(config, sub_df, ngram_dicts):
    return MultiChannelDataset(
        texts=sub_df["Name"], labels=sub_df["NOC"],
        lang_labels=sub_df["LanguageGroup"],
        uni_ngram_dict=ngram_dicts["uni"], bi_ngram_dict=ngram_dicts["bi"],
        tri_ngram_dict=ngram_dicts["tri"],
        max_length=config["data"]["max_length"],
        country_idx=get_country_idx(config),
        lang_group_idx=get_lang_group_idx(config),
    )


def load_preprocessed_df(config):
    df = get_loaded_df(config).copy()
    df["Name"] = df["Name"].apply(replace_disallowed_chars)
    return df[df["Name"].str.len() > 0].reset_index(drop=True)


@torch.no_grad()
def evaluate_test(model, loader, device):
    """nat top1/top5/macro-F1 + lang top1/top3/macro-F1 (Trainer.evaluate 미러)."""
    nat_top5_hits, lang_top3_hits = [], []
    nat_preds, nat_trues, lang_preds, lang_trues = [], [], [], []
    for batch in loader:
        *inputs, nat_t, lang_t = batch
        nat_logits, lang_logits = model(*[t.to(device) for t in inputs])
        nat_top5 = nat_logits.topk(5, dim=1).indices.cpu()
        lang_top3 = lang_logits.topk(3, dim=1).indices.cpu()
        nat_preds.extend(nat_top5[:, 0].tolist())
        lang_preds.extend(lang_top3[:, 0].tolist())
        nat_top5_hits.extend((nat_top5 == nat_t.unsqueeze(1)).any(dim=1).tolist())
        lang_top3_hits.extend((lang_top3 == lang_t.unsqueeze(1)).any(dim=1).tolist())
        nat_trues.extend(nat_t.tolist())
        lang_trues.extend(lang_t.tolist())
    nat_preds, nat_trues = np.array(nat_preds), np.array(nat_trues)
    lang_preds, lang_trues = np.array(lang_preds), np.array(lang_trues)
    return {
        "nat_top1": float((nat_preds == nat_trues).mean() * 100),
        "nat_top5": float(np.mean(nat_top5_hits) * 100),
        "nat_macro_f1": float(f1_score(nat_trues, nat_preds, average="macro",
                                       zero_division=0) * 100),
        "lang_top1": float((lang_preds == lang_trues).mean() * 100),
        "lang_top3": float(np.mean(lang_top3_hits) * 100),
        "lang_macro_f1": float(f1_score(lang_trues, lang_preds, average="macro",
                                        zero_division=0) * 100),
    }


def compute_rho(model, full_loader, country_list, dicl_cnl, device, n_boot):
    embeddings, nat_labels, _ = extract_embeddings(model, full_loader, device)
    centroids = compute_country_centroids(embeddings, nat_labels, country_list)
    rows = []
    for (noc_i, noc_j), dist in compute_pairwise_distances(centroids, country_list).items():
        key = tuple(sorted([noc_to_iso3(noc_i), noc_to_iso3(noc_j)]))
        if key in dicl_cnl:
            rows.append({"emb_sim": 1.0 - dist, "cnl": dicl_cnl[key]})
    pairs = pd.DataFrame(rows)
    sub = pairs[pairs["cnl"] > 0]
    rho_all = bootstrap_rho_ci(pairs["cnl"].values, pairs["emb_sim"].values, n_boot=n_boot)
    rho_sub = bootstrap_rho_ci(sub["cnl"].values, sub["emb_sim"].values, n_boot=n_boot)
    return {
        "cnl_rho_all": round(rho_all["rho"], 4),
        "cnl_rho_all_ci": f'[{rho_all["ci_low"]:.3f}, {rho_all["ci_high"]:.3f}]',
        "cnl_rho_subset": round(rho_sub["rho"], 4),
        "cnl_rho_subset_ci": f'[{rho_sub["ci_low"]:.3f}, {rho_sub["ci_high"]:.3f}]',
        "cnl_n_pairs": int(len(pairs)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=os.path.join("configs", "base_config.yaml"))
    parser.add_argument("--alphas", nargs="+", type=float, default=[0.0])
    parser.add_argument("--n_boot", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=512)
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        config = yaml.safe_load(f)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    country_list = get_country_list(config)
    dicl_cnl = load_dicl(DICL_PATH, metric="cnl")
    ngram_dicts = get_ngram_dicts("multi")

    # 전체 데이터 loader (ρ 계산용 — split 무관이라 1회 구축, 전 모델 공유)
    full_df = load_preprocessed_df(config)
    full_loader = DataLoader(build_dataset(config, full_df, ngram_dicts),
                             batch_size=args.batch_size, shuffle=False)
    print(f"[Data] full={len(full_df)}  NOCs={len(country_list)}")

    rows = []
    for alpha in args.alphas:
        for seed in SEEDS:
            ckpt = os.path.join(CKPT_DIR, f"multiseed_alpha_{alpha}_seed{seed}.pt")
            if not os.path.exists(ckpt):
                print(f"[SKIP] 없음: {ckpt}")
                continue

            t0 = time.time()
            # 시드별 자기 test split 재현 (set_seed 직후 robust_split — 원 실험 순서)
            set_seed(seed)
            _, _, test_idx = robust_split(full_df)
            test_loader = DataLoader(
                build_dataset(config, full_df.iloc[test_idx], ngram_dicts),
                batch_size=args.batch_size, shuffle=False)

            model = load_model(config, CONFORMER_OVERRIDES, ckpt, device)
            metrics = evaluate_test(model, test_loader, device)
            rho = compute_rho(model, full_loader, country_list, dicl_cnl,
                              device, args.n_boot)
            del model
            torch.cuda.empty_cache()

            rows.append({"alpha": alpha, "seed": seed, **metrics, **rho,
                         "checkpoint": ckpt})
            print(f"[alpha={alpha} seed={seed}] "
                  f"nat_top1={metrics['nat_top1']:.2f}  "
                  f"nat_macro_f1={metrics['nat_macro_f1']:.2f}  "
                  f"rho_all={rho['cnl_rho_all']}  rho_sub={rho['cnl_rho_subset']}  "
                  f"({time.time() - t0:.0f}s)")

    result_df = pd.DataFrame(rows)
    out_csv = os.path.join(OUTPUT_DIR, "alpha_multiseed_results.csv")
    result_df.to_csv(out_csv, index=False)

    print("\n===== α ablation multi-seed 요약 (mean ± std over seeds) =====")
    for alpha, g in result_df.groupby("alpha"):
        line = f"alpha={alpha}: "
        for col in ["nat_top1", "nat_macro_f1", "lang_top1",
                    "cnl_rho_all", "cnl_rho_subset"]:
            line += f"{col}={g[col].mean():.3f}±{g[col].std():.3f}  "
        print(line + f"(n={len(g)} seeds)")
    print(f"\n저장: {out_csv}")


if __name__ == "__main__":
    main()
