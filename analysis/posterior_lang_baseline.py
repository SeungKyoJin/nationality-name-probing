"""
analysis/posterior_lang_baseline.py

국적 posterior 집계 baseline (rebuttal R2, Reviewer HUuH Comments 대응).

HUuH: "168-way 국적 posterior를 39개 언어권으로 합산하는 단순 baseline이
언어권 head의 추가 가치를 판단하는 데 도움이 될 것."

방법:
  · Conformer 3시드 checkpoint, 각자 자기 test split (confusion_asymmetry와 동일 절차)
  · 방법 A (aggregation): softmax(nat_logits) → NOC→LanguageGroup 매핑으로 그룹별 합산
    → argmax  (언어권 head를 전혀 쓰지 않음)
  · 방법 B (lang head): 기존 lang_logits argmax
  · 두 방법의 lang Top-1 / macro-F1을 시드별 비교

산출: outputs/analysis/posterior_lang_baseline/results.csv (+콘솔 mean±std)

실행:
  python -m analysis.posterior_lang_baseline --config configs/base_config.yaml
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

from data.data_loader import (
    get_loaded_df, get_country_idx, get_country_list, get_lang_group_idx,
    get_ngram_dicts, robust_split,
)
from data.dataset import MultiChannelDataset
from run_table1_rho_ci import load_model
from utils.util import replace_disallowed_chars, set_seed

CKPT_ROOT = "c:/Users/t4716/PycharmProjects/nationality_classifier/outputs/checkpoints"
CONFORMER_SEEDS = {
    42: "260516/multiseed_ours_alpha03_seed42.pt",
    7: "260516/multiseed_ours_alpha03_seed7.pt",
    123: "260517/multiseed_ours_alpha03_seed123.pt",
}
CONFORMER_OVERRIDES = {
    "type": "multichannel_conformer",
    "channel_ensemble_head": True,
    "channel_fusion_type": "uniform",
}
OUTPUT_DIR = os.path.join("outputs", "analysis", "posterior_lang_baseline")


def build_noc_to_lang(config, country_idx, lang_group_idx):
    """NOC 인덱스 → LanguageGroup 인덱스 매핑 벡터 [168]."""
    raw = pd.read_csv(config["data"]["raw_path"], usecols=["NOC", "LanguageGroup"])
    mapping = raw.drop_duplicates("NOC").set_index("NOC")["LanguageGroup"].to_dict()
    noc_to_lang = np.zeros(len(country_idx), dtype=np.int64)
    for noc, i in country_idx.items():
        noc_to_lang[i] = lang_group_idx[mapping[noc]]
    return noc_to_lang


def build_test_loader(config, batch_size):
    df = get_loaded_df(config).copy()
    df["Name"] = df["Name"].apply(replace_disallowed_chars)
    df = df[df["Name"].str.len() > 0].reset_index(drop=True)
    _, _, test_idx = robust_split(df)
    test_df = df.iloc[test_idx]
    ngram_dicts = get_ngram_dicts("multi")
    dataset = MultiChannelDataset(
        texts=test_df["Name"], labels=test_df["NOC"],
        lang_labels=test_df["LanguageGroup"],
        uni_ngram_dict=ngram_dicts["uni"], bi_ngram_dict=ngram_dicts["bi"],
        tri_ngram_dict=ngram_dicts["tri"],
        max_length=config["data"]["max_length"],
        country_idx=get_country_idx(config),
        lang_group_idx=get_lang_group_idx(config),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)


@torch.no_grad()
def predict_both(model, loader, noc_to_lang_t, device):
    """(집계 lang 예측, head lang 예측, lang 정답) — test 전체."""
    agg_preds, head_preds, lang_trues = [], [], []
    for batch in loader:
        *inputs, _, lang_t = batch
        nat_logits, lang_logits = model(*[t.to(device) for t in inputs])
        # 방법 A: nat posterior를 그룹별 합산 (index_add)
        nat_probs = nat_logits.softmax(dim=1)
        group_probs = torch.zeros(
            nat_probs.size(0), lang_logits.size(1), device=device)
        group_probs.index_add_(1, noc_to_lang_t, nat_probs)
        agg_preds.extend(group_probs.argmax(dim=1).cpu().tolist())
        # 방법 B: lang head
        head_preds.extend(lang_logits.argmax(dim=1).cpu().tolist())
        lang_trues.extend(lang_t.tolist())
    return np.array(agg_preds), np.array(head_preds), np.array(lang_trues)


def lang_metrics(preds, trues, n_groups):
    return {
        "top1": float((preds == trues).mean() * 100),
        "macro_f1": float(f1_score(trues, preds, labels=range(n_groups),
                                   average="macro", zero_division=0) * 100),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=os.path.join("configs", "base_config.yaml"))
    p.add_argument("--batch_size", type=int, default=512)
    args = p.parse_args()

    with open(args.config, encoding="utf-8") as f:
        config = yaml.safe_load(f)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    country_idx = get_country_idx(config)
    lang_group_idx = get_lang_group_idx(config)
    n_groups = len(lang_group_idx)
    noc_to_lang = build_noc_to_lang(config, country_idx, lang_group_idx)
    noc_to_lang_t = torch.from_numpy(noc_to_lang).to(device)

    rows = []
    for seed, rel_path in CONFORMER_SEEDS.items():
        set_seed(seed)                          # 시드별 자기 split 재현
        loader = build_test_loader(config, args.batch_size)
        model = load_model(config, CONFORMER_OVERRIDES,
                           os.path.join(CKPT_ROOT, rel_path), device)
        agg_preds, head_preds, trues = predict_both(
            model, loader, noc_to_lang_t, device)
        del model
        torch.cuda.empty_cache()

        agg = lang_metrics(agg_preds, trues, n_groups)
        head = lang_metrics(head_preds, trues, n_groups)
        agree = float((agg_preds == head_preds).mean() * 100)
        rows.append({"seed": seed,
                     "agg_top1": agg["top1"], "agg_macro_f1": agg["macro_f1"],
                     "head_top1": head["top1"], "head_macro_f1": head["macro_f1"],
                     "pred_agreement": agree})
        print(f"[seed {seed}] aggregation: top1={agg['top1']:.2f} mF1={agg['macro_f1']:.2f}"
              f"  |  lang head: top1={head['top1']:.2f} mF1={head['macro_f1']:.2f}"
              f"  |  두 방법 예측 일치율 {agree:.1f}%")

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUTPUT_DIR, "results.csv"), index=False)
    print("\n===== 3-seed mean ± std =====")
    for col in ["agg_top1", "agg_macro_f1", "head_top1", "head_macro_f1"]:
        print(f"{col:14s}: {df[col].mean():.2f} ± {df[col].std():.2f}")
    print(f"저장: {OUTPUT_DIR}/results.csv")


if __name__ == "__main__":
    main()
