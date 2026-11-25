"""
analysis/confusion_asymmetry.py

Confusion matrix 기반 비대칭 오분류 진단 (rebuttal R2, Reviewer HUuH 대응).

HUuH 지적 3건을 한 실험으로 검증한다:
  1. "isolation score는 국가쌍 간 비대칭 confusion을 못 잡는다"
     → high-CNL 쌍의 양방향 오분류율을 직접 측정해 비대칭의 실재와 크기를 보고.
  2. "비대칭의 방향은 무엇이 정하는가"
     → 가설: 오분류는 train 데이터가 적은 클래스 → 많은 클래스로 흐른다.
       (Wilcoxon signed-rank + 방향 일치 비율)
  3. "Australia가 shared-language 이웃 때문에 어렵다는 주장은 검증 불가"
     → F1 하위 10개국의 top-3 오분류 목적지 + 해당 쌍 CNL 표.

방법:
  · 논문 Conformer 3시드 checkpoint (multiseed_ours_alpha03_seed{42,7,123})
  · ⚠ multi-seed 실험은 시드가 robust_split까지 바꾸므로, 각 checkpoint를
    "자기 시드의 test split"에서 평가해야 한다 (seed-42 split에 seed-7 모델을
    올리면 train 데이터가 test에 섞여 leakage → 정확도 ~60%로 부풀려짐).
  · 시드별 row-normalized confusion rate를 계산한 뒤 3시드 평균.
  · sanity check: 시드별 재계산 per-class F1 ↔ 그 시드의 저장된 class_report 대조.

산출 (outputs/analysis/confusion_asymmetry/):
  · confusion_rate_nat.csv          — 행 정규화 오분류율 (3시드 평균)
  · high_cnl_pair_asymmetry.csv     — CNL≥threshold 쌍의 양방향 rate + train counts
  · worst10_confusion_targets.csv   — F1 하위 10개국 top-3 목적지
  · summary.json                    — Wilcoxon 결과 + 방향 일치 비율

실행:
  python -m analysis.confusion_asymmetry --config configs/base_config.yaml
"""

from __future__ import annotations

import argparse
import json
import os
from itertools import combinations

import numpy as np
import pandas as pd
import torch
import yaml
from scipy.stats import wilcoxon
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

from analysis.failure_modes import load_class_report, get_noc_total_counts
from analysis.noc_iso3_mapping import noc_to_iso3
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
SANITY_REPORTS = {
    42: os.path.join("outputs", "checkpoints", "260516",
                     "class_report_nat_multiseed_ours_alpha03_seed42.pt.csv"),
    7: os.path.join("outputs", "checkpoints", "260516",
                    "class_report_nat_multiseed_ours_alpha03_seed7.pt.csv"),
    123: os.path.join("outputs", "checkpoints", "260517",
                      "class_report_nat_multiseed_ours_alpha03_seed123.pt.csv"),
}
# 원본 평가는 AMP(autocast)라 borderline 샘플 1개 수준의 argmax 차이가 가능.
# per-class F1 기준 샘플 1개 flip ≈ 0.003~0.01 → 0.01을 허용 오차로 둔다.
SANITY_F1_TOLERANCE = 0.01


def build_test_loader(config, batch_size):
    """seed-42 robust_split 재현 → test split의 MultiChannelDataset loader."""
    df = get_loaded_df(config).copy()
    df["Name"] = df["Name"].apply(replace_disallowed_chars)
    df = df[df["Name"].str.len() > 0].reset_index(drop=True)

    train_idx, _, test_idx = robust_split(df)
    train_counts = df.iloc[train_idx]["NOC"].value_counts().to_dict()
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
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return loader, train_counts


@torch.no_grad()
def predict_nat(model, loader, device):
    """test set 전체의 (true, pred) 국적 인덱스."""
    trues, preds = [], []
    for batch in loader:
        *inputs, nat_target, _ = batch
        nat_logits, _ = model(*[t.to(device) for t in inputs])
        preds.extend(nat_logits.argmax(dim=1).cpu().tolist())
        trues.extend(nat_target.tolist())
    return np.array(trues), np.array(preds)


def confusion_counts(trues, preds, n_classes):
    counts = np.zeros((n_classes, n_classes), dtype=np.float64)
    np.add.at(counts, (trues, preds), 1.0)
    return counts


def sanity_check(seed, trues, preds, country_list):
    """재계산 per-class F1 ↔ 그 시드의 저장된 class report 대조 (split·모델 검증)."""
    recomputed = f1_score(
        trues, preds, labels=range(len(country_list)), average=None,
        zero_division=0,
    )
    saved = load_class_report(SANITY_REPORTS[seed]).set_index("NOC")["f1-score"]
    diffs = [
        abs(recomputed[i] - saved[noc])
        for i, noc in enumerate(country_list) if noc in saved.index
    ]
    max_diff = max(diffs)
    top1 = float((trues == preds).mean() * 100)
    status = "일치" if max_diff < SANITY_F1_TOLERANCE else "불일치!"
    print(f"[Sanity seed {seed}] Nat-T1={top1:.2f}  "
          f"per-class F1 max|Δ|={max_diff:.2e} vs 저장 리포트 → {status}")
    if max_diff >= SANITY_F1_TOLERANCE:
        raise RuntimeError(
            f"seed {seed}: split 재현 실패 (max|Δ|={max_diff:.4f}) — "
            "leakage 위험이 있으므로 중단")
    return max_diff


def main():
    p = argparse.ArgumentParser(description="비대칭 confusion 진단")
    p.add_argument("--config", default=os.path.join("configs", "base_config.yaml"))
    p.add_argument("--dicl_path",
                   default=os.path.join("data", "linguistics", "DICL_v2.csv"))
    p.add_argument("--output_dir",
                   default=os.path.join("outputs", "analysis", "confusion_asymmetry"))
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--cnl_threshold", type=float, default=0.5,
                   help="비대칭 분석 대상 쌍의 최소 CNL")
    p.add_argument("--min_total", type=int, default=500,
                   help="분석 대상 NOC 최소 total sample (paper §5와 동일)")
    p.add_argument("--cap", type=int, default=3000)
    p.add_argument("--size_ratio_min", type=float, default=1.1,
                   help="방향 가설 검정에서 '크기가 다르다'로 볼 train count 비율")
    args = p.parse_args()

    with open(args.config, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.output_dir, exist_ok=True)

    country_list = get_country_list(config)
    n_classes = len(country_list)

    # ── 시드별: 자기 split 재현 → 예측 → row-normalized rate ──
    rate_seeds, per_class_f1_seeds, train_counts_seeds = [], [], []
    for seed, rel_path in CONFORMER_SEEDS.items():
        set_seed(seed)                       # robust_split 직전 시드 고정 (원 실험과 동일 순서)
        loader, train_counts = build_test_loader(config, args.batch_size)
        print(f"[seed {seed}] test={len(loader.dataset)}")

        model = load_model(config, CONFORMER_OVERRIDES,
                           os.path.join(CKPT_ROOT, rel_path), device)
        trues, preds = predict_nat(model, loader, device)
        del model
        torch.cuda.empty_cache()

        sanity_check(seed, trues, preds, country_list)

        counts = confusion_counts(trues, preds, n_classes)
        support = counts.sum(axis=1)
        rate_seeds.append(np.divide(
            counts, support[:, None],
            out=np.zeros_like(counts), where=support[:, None] > 0))
        per_class_f1_seeds.append(f1_score(
            trues, preds, labels=range(n_classes), average=None, zero_division=0))
        train_counts_seeds.append(train_counts)

    rate = np.mean(rate_seeds, axis=0)
    seed_mean_f1 = np.mean(per_class_f1_seeds, axis=0)
    # train count는 stratified split이라 시드 간 거의 동일 — 평균 사용
    train_counts = {
        noc: int(np.mean([tc.get(noc, 0) for tc in train_counts_seeds]))
        for noc in country_list
    }

    rate_df = pd.DataFrame(rate, index=country_list, columns=country_list)
    rate_df.to_csv(os.path.join(args.output_dir, "confusion_rate_nat.csv"))

    # ── DICL CNL 쌍 lookup (mirrored CSV → 양방향 키) ──
    dicl = pd.read_csv(args.dicl_path, usecols=["iso3_i", "iso3_j", "cnl"])
    cnl_lookup = dict(zip(zip(dicl["iso3_i"], dicl["iso3_j"]), dicl["cnl"]))

    def pair_cnl(noc_a, noc_b):
        key = (noc_to_iso3(noc_a), noc_to_iso3(noc_b))
        return cnl_lookup.get(key, cnl_lookup.get((key[1], key[0])))

    # ── 분석 대상 NOC: total ≥ min_total (paper §5의 112개국 필터와 동일) ──
    noc_totals = get_noc_total_counts(config["data"]["raw_path"], cap=args.cap)
    idx_of = {noc: i for i, noc in enumerate(country_list)}
    eligible = [noc for noc in country_list
                if noc_totals.get(noc, 0) >= args.min_total]
    print(f"[Filter] total ≥ {args.min_total} → {len(eligible)} NOCs")

    # ── high-CNL 쌍의 양방향 오분류율 ──
    pair_rows = []
    for noc_a, noc_b in combinations(eligible, 2):
        cnl = pair_cnl(noc_a, noc_b)
        if cnl is None or cnl < args.cnl_threshold:
            continue
        i, j = idx_of[noc_a], idx_of[noc_b]
        n_train_a = train_counts.get(noc_a, 0)
        n_train_b = train_counts.get(noc_b, 0)
        # small → 항상 train count가 작은 쪽 (방향 가설의 기준 정렬)
        if n_train_a <= n_train_b:
            small, large, r_sl, r_ls = noc_a, noc_b, rate[i, j], rate[j, i]
            n_small, n_large = n_train_a, n_train_b
        else:
            small, large, r_sl, r_ls = noc_b, noc_a, rate[j, i], rate[i, j]
            n_small, n_large = n_train_b, n_train_a
        pair_rows.append({
            "small": small, "large": large, "cnl": cnl,
            "n_train_small": n_small, "n_train_large": n_large,
            "rate_small_to_large": r_sl, "rate_large_to_small": r_ls,
            "asymmetry": r_sl - r_ls,
            "size_ratio": n_large / max(n_small, 1),
        })
    pairs_df = pd.DataFrame(pair_rows).sort_values("asymmetry", ascending=False)
    pairs_df.to_csv(
        os.path.join(args.output_dir, "high_cnl_pair_asymmetry.csv"), index=False)

    # ── 방향 가설 검정: train count가 실제로 다른 쌍에서만 ──
    testable = pairs_df[pairs_df["size_ratio"] >= args.size_ratio_min]
    equal_size = pairs_df[pairs_df["size_ratio"] < args.size_ratio_min]
    stat, p_value = wilcoxon(
        testable["rate_small_to_large"], testable["rate_large_to_small"],
        alternative="greater",
    )
    direction_agree = float(
        (testable["rate_small_to_large"] > testable["rate_large_to_small"]).mean())

    summary = {
        "n_high_cnl_pairs": int(len(pairs_df)),
        "cnl_threshold": args.cnl_threshold,
        "n_testable_pairs (train size differs)": int(len(testable)),
        "n_equal_size_pairs": int(len(equal_size)),
        "wilcoxon": {"statistic": float(stat), "p_one_sided": float(p_value),
                     "alternative": "small→large > large→small"},
        "direction_agreement": direction_agree,
        "mean_rate_small_to_large": float(testable["rate_small_to_large"].mean()),
        "mean_rate_large_to_small": float(testable["rate_large_to_small"].mean()),
        "equal_size_mean_asymmetry_abs": float(equal_size["asymmetry"].abs().mean())
        if len(equal_size) else None,
    }

    # ── F1 하위 10개국의 top-3 오분류 목적지 ──
    worst_rows = []
    eligible_idx = [(noc, idx_of[noc]) for noc in eligible]
    worst10 = sorted(eligible_idx, key=lambda t: seed_mean_f1[t[1]])[:10]
    for noc, i in worst10:
        row_rate = rate[i].copy()
        row_rate[i] = 0.0                      # 자기 자신(정답) 제외
        top3 = np.argsort(row_rate)[::-1][:3]
        entry = {"NOC": noc, "f1_seed_mean": round(float(seed_mean_f1[i]), 3),
                 "self_rate": round(float(rate[i, i]), 3),
                 "n_train": train_counts.get(noc, 0)}
        for rank, j in enumerate(top3, 1):
            dest = country_list[j]
            entry[f"dest{rank}"] = dest
            entry[f"dest{rank}_rate"] = round(float(row_rate[j]), 3)
            entry[f"dest{rank}_cnl"] = pair_cnl(noc, dest)
            entry[f"dest{rank}_n_train"] = train_counts.get(dest, 0)
        worst_rows.append(entry)
    worst_df = pd.DataFrame(worst_rows)
    worst_df.to_csv(
        os.path.join(args.output_dir, "worst10_confusion_targets.csv"), index=False)

    with open(os.path.join(args.output_dir, "summary.json"), "w",
              encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n===== 비대칭 confusion 요약 =====")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print("\n===== 비대칭 상위 10쌍 (small→large 기준) =====")
    print(pairs_df.head(10).to_string(index=False))
    print("\n===== F1 하위 10개국 top-3 오분류 목적지 =====")
    print(worst_df.to_string(index=False))
    print(f"\n저장: {args.output_dir}/")


if __name__ == "__main__":
    main()
