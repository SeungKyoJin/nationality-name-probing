"""
CANINE-s reference 실험 (rebuttal R1, Reviewer U6VP 대응).

pretrained multilingual character-level encoder를 동일 프로토콜로 fine-tune해서
Table 1의 "reference row" 후보를 만든다. from-scratch 모델들과 달리
사전학습이 corpus-level 국가-언어 지식을 주입하므로 baseline이 아닌
reference point로만 보고한다 (본문 주장과의 관계는 rebuttal_r1.md 참조).

프로토콜 미러링:
  - seed 42 + robust_split → 기존 실험과 동일한 train/val/test 분할
  - 동일 전처리 (lowercase + a-z/space whitelist)
  - hierarchical loss L = α·L_lang + (1-α)·L_nat, α=0.3
  - grad clipping 1.0, early stopping (val loss), AMP
  - cnl ρ: 전체 데이터 임베딩 → NOC centroid → cosine → Spearman
    (analysis/dicl_correlation.compute_cnl_rho_quick과 동일 절차)

실행:
  pip install transformers  # 최초 1회
  python run_canine_reference.py --config configs/base_config.yaml
"""

import os
import argparse
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader, Dataset

from data.data_loader import (
    get_loaded_df, get_country_idx, get_country_list, get_lang_group_idx,
    robust_split,
)
from utils.util import set_seed, replace_disallowed_chars
from analysis.dicl_correlation import (
    load_dicl, compute_country_centroids, compute_pairwise_distances,
)
from analysis.noc_iso3_mapping import noc_to_iso3
from sklearn.metrics import f1_score

MODEL_NAME = "google/canine-s"
OUTPUT_DIR = os.path.join("outputs", "canine_reference")
DICL_PATH = os.path.join("data", "linguistics", "DICL_v2.csv")

# fine-tune 하이퍼파라미터 (pretrained encoder 관례값 — from-scratch config와 다름)
FT_LR = 2e-5
FT_BATCH_SIZE = 64
FT_MAX_EPOCHS = 6
FT_PATIENCE = 2
GRAD_CLIP_MAX_NORM = 1.0
# CANINE special token([CLS]/[SEP]) 2개를 더해 기존 max_length=90과 문자 수를 맞춤
TOKENIZER_EXTRA_TOKENS = 2


class NameDataset(Dataset):
    """전처리된 이름 문자열 → CANINE 토큰 (uni/bi/tri 채널 없이 raw 문자열)."""

    def __init__(self, names, nat_labels, lang_labels, tokenizer, max_length):
        encoded = tokenizer(
            list(names), padding="max_length", truncation=True,
            max_length=max_length, return_tensors="pt",
        )
        self.input_ids = encoded["input_ids"]
        self.attention_mask = encoded["attention_mask"]
        self.nat_labels = torch.tensor(list(nat_labels), dtype=torch.long)
        self.lang_labels = torch.tensor(list(lang_labels), dtype=torch.long)

    def __len__(self):
        return len(self.nat_labels)

    def __getitem__(self, i):
        return (self.input_ids[i], self.attention_mask[i],
                self.nat_labels[i], self.lang_labels[i])


class CanineDualHead(nn.Module):
    """CANINE encoder + 기존 모델과 동일한 dual head (nat 168 / lang 39)."""

    def __init__(self, model_name, n_classes, n_lang_groups, dropout=0.1):
        super().__init__()
        from transformers import CanineModel
        self.encoder = CanineModel.from_pretrained(model_name)
        hidden = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.head_nat = nn.Linear(hidden, n_classes)
        self.head_lang = nn.Linear(hidden, n_lang_groups)

    def forward(self, input_ids, attention_mask):
        pooled = self.encoder(
            input_ids=input_ids, attention_mask=attention_mask
        ).pooler_output
        pooled = self.dropout(pooled)
        return self.head_nat(pooled), self.head_lang(pooled)

    @torch.no_grad()
    def encode(self, input_ids, attention_mask):
        """cnl ρ 분석용 pooled representation (기존 모델들의 encode()와 동일 역할)."""
        return self.encoder(
            input_ids=input_ids, attention_mask=attention_mask
        ).pooler_output


def build_splits(config, tokenizer):
    """기존 파이프라인과 동일한 전처리·분할로 Dataset 3개 + 전체 Dataset 생성."""
    df = get_loaded_df(config).copy()
    df["Name"] = df["Name"].apply(replace_disallowed_chars)
    df = df[df["Name"].str.len() > 0].reset_index(drop=True)

    country_idx = get_country_idx(config)
    lang_group_idx = get_lang_group_idx(config)
    df["nat_idx"] = df["NOC"].map(country_idx)
    df["lang_idx"] = df["LanguageGroup"].map(lang_group_idx)

    max_length = config["data"]["max_length"] + TOKENIZER_EXTRA_TOKENS

    # robust_split은 np.random.shuffle을 쓰므로 호출 직전 시드 상태가 동일해야
    # 기존 실험과 같은 분할이 나온다 (run_all_experiments.py와 동일 순서).
    train_idx, val_idx, test_idx = robust_split(df)

    def make(sub_df):
        return NameDataset(
            sub_df["Name"], sub_df["nat_idx"], sub_df["lang_idx"],
            tokenizer, max_length,
        )

    return (
        make(df.iloc[train_idx]),
        make(df.iloc[val_idx]),
        make(df.iloc[test_idx]),
        make(df),          # cnl ρ는 전체 데이터 기준 (compute_cnl_rho_quick과 동일)
        df,
    )


def hierarchical_loss(nat_logits, lang_logits, nat_target, lang_target, alpha):
    ce = nn.functional.cross_entropy
    return alpha * ce(lang_logits, lang_target) + (1 - alpha) * ce(nat_logits, nat_target)


@torch.no_grad()
def evaluate(model, loader, device, alpha):
    """Trainer.evaluate()와 동일한 지표 계산 (nat top1/5, macro-F1 / lang top1/3, macro-F1)."""
    model.eval()
    total_loss, total_n = 0.0, 0
    nat_preds, nat_top5_hits, nat_targets = [], [], []
    lang_preds, lang_top3_hits, lang_targets = [], [], []

    for input_ids, attention_mask, nat_t, lang_t in loader:
        input_ids, attention_mask = input_ids.to(device), attention_mask.to(device)
        nat_t_dev, lang_t_dev = nat_t.to(device), lang_t.to(device)

        nat_logits, lang_logits = model(input_ids, attention_mask)
        loss = hierarchical_loss(nat_logits, lang_logits, nat_t_dev, lang_t_dev, alpha)
        total_loss += loss.item() * nat_t.size(0)
        total_n += nat_t.size(0)

        nat_top5 = nat_logits.topk(5, dim=1).indices.cpu()
        lang_top3 = lang_logits.topk(3, dim=1).indices.cpu()
        nat_preds.extend(nat_top5[:, 0].tolist())
        lang_preds.extend(lang_top3[:, 0].tolist())
        nat_top5_hits.extend((nat_top5 == nat_t.unsqueeze(1)).any(dim=1).tolist())
        lang_top3_hits.extend((lang_top3 == lang_t.unsqueeze(1)).any(dim=1).tolist())
        nat_targets.extend(nat_t.tolist())
        lang_targets.extend(lang_t.tolist())

    nat_preds, nat_targets = np.array(nat_preds), np.array(nat_targets)
    lang_preds, lang_targets = np.array(lang_preds), np.array(lang_targets)
    return {
        "loss": total_loss / max(total_n, 1),
        "nat_top1": float((nat_preds == nat_targets).mean() * 100),
        "nat_top5": float(np.mean(nat_top5_hits) * 100),
        "nat_macro_f1": float(f1_score(nat_targets, nat_preds, average="macro") * 100),
        "lang_top1": float((lang_preds == lang_targets).mean() * 100),
        "lang_top3": float(np.mean(lang_top3_hits) * 100),
        "lang_macro_f1": float(f1_score(lang_targets, lang_preds, average="macro") * 100),
    }


@torch.no_grad()
def compute_cnl_rho(model, full_loader, df, country_list, device):
    """전체 데이터 임베딩 → NOC centroid → cosine → CNL Spearman ρ (all / subset)."""
    from scipy.stats import spearmanr

    model.eval()
    chunks = []
    for input_ids, attention_mask, _, _ in full_loader:
        emb = model.encode(input_ids.to(device), attention_mask.to(device))
        chunks.append(emb.cpu().numpy())
    embeddings = np.concatenate(chunks, axis=0)
    nat_labels = df["nat_idx"].values

    centroids = compute_country_centroids(embeddings, nat_labels, country_list)
    dicl_cnl = load_dicl(DICL_PATH, metric="cnl")

    pair_rows = []
    for (noc_i, noc_j), emb_dist in compute_pairwise_distances(centroids, country_list).items():
        key = tuple(sorted([noc_to_iso3(noc_i), noc_to_iso3(noc_j)]))
        if key in dicl_cnl:
            pair_rows.append({
                "noc_i": noc_i, "noc_j": noc_j,
                "emb_sim": 1.0 - emb_dist, "cnl": dicl_cnl[key],
            })

    pairs = pd.DataFrame(pair_rows)
    rho_all, _ = spearmanr(pairs["cnl"], pairs["emb_sim"])
    subset = pairs[pairs["cnl"] > 0]
    rho_sub, _ = spearmanr(subset["cnl"], subset["emb_sim"])
    return {
        "cnl_rho_all": round(float(rho_all), 4),
        "cnl_rho_subset": round(float(rho_sub), 4),
        "cnl_n_pairs": len(pairs),
    }, pairs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base_config.yaml")
    parser.add_argument("--model_name", default=MODEL_NAME)
    parser.add_argument("--epochs", type=int, default=FT_MAX_EPOCHS)
    parser.add_argument("--batch_size", type=int, default=FT_BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=FT_LR)
    args = parser.parse_args()

    from transformers import CanineTokenizer

    with open(args.config, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    alpha = config["train"]["alpha"]
    set_seed(config["experiment"]["seed"])   # robust_split 직전 시드 고정 (분할 재현)

    tokenizer = CanineTokenizer.from_pretrained(args.model_name)
    train_ds, val_ds, test_ds, full_ds, df = build_splits(config, tokenizer)
    print(f"[Data] train={len(train_ds)}  val={len(val_ds)}  test={len(test_ds)}")

    dl_kwargs = dict(num_workers=2, pin_memory=True)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, **dl_kwargs)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size * 2, **dl_kwargs)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size * 2, **dl_kwargs)
    full_loader = DataLoader(full_ds, batch_size=args.batch_size * 2, **dl_kwargs)

    model = CanineDualHead(
        args.model_name,
        n_classes=config["model"]["num_classes"],
        n_lang_groups=config["model"]["num_lang_groups"],
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=config["train"]["weight_decay"])
    scaler = torch.amp.GradScaler(device)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    ckpt_path = os.path.join(OUTPUT_DIR, "canine_s_dual.pt")
    best_val_loss, patience_left = float("inf"), FT_PATIENCE

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss, n_seen, t0 = 0.0, 0, time.time()
        for input_ids, attention_mask, nat_t, lang_t in train_loader:
            input_ids, attention_mask = input_ids.to(device), attention_mask.to(device)
            nat_t, lang_t = nat_t.to(device), lang_t.to(device)

            optimizer.zero_grad()
            with torch.amp.autocast(device):
                nat_logits, lang_logits = model(input_ids, attention_mask)
                loss = hierarchical_loss(nat_logits, lang_logits, nat_t, lang_t, alpha)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_MAX_NORM)
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += loss.item() * nat_t.size(0)
            n_seen += nat_t.size(0)

        val_metrics = evaluate(model, val_loader, device, alpha)
        print(f"[Epoch {epoch}] train_loss={epoch_loss / n_seen:.4f}  "
              f"val_loss={val_metrics['loss']:.4f}  "
              f"val_nat_top1={val_metrics['nat_top1']:.2f}  "
              f"val_lang_top1={val_metrics['lang_top1']:.2f}  "
              f"({time.time() - t0:.0f}s)")

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            patience_left = FT_PATIENCE
            torch.save(model.state_dict(), ckpt_path)
        else:
            patience_left -= 1
            if patience_left == 0:
                print(f"[EarlyStopping] epoch {epoch}에서 종료 (best val_loss={best_val_loss:.4f})")
                break

    # best checkpoint 복원 후 test + cnl ρ
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    test_metrics = evaluate(model, test_loader, device, alpha)
    country_list = get_country_list(config)
    rho_metrics, pairs = compute_cnl_rho(model, full_loader, df, country_list, device)

    result = {
        "model": args.model_name, "alpha": alpha, "lr": args.lr,
        "batch_size": args.batch_size, **test_metrics, **rho_metrics,
        "timestamp": pd.Timestamp.now().isoformat(),
    }
    results_path = os.path.join(OUTPUT_DIR, "results.csv")
    pd.DataFrame([result]).to_csv(
        results_path, mode="a", index=False, header=not os.path.exists(results_path),
    )
    # 페어별 유사도 저장 → 기존 모델과의 paired bootstrap에 사용
    pairs.to_csv(os.path.join(OUTPUT_DIR, "pair_similarities.csv"), index=False)

    print("\n===== CANINE-s reference 결과 =====")
    for k, v in result.items():
        print(f"  {k}: {v}")
    print(f"\n결과 저장: {results_path}")


if __name__ == "__main__":
    main()
