"""
CANINE-s reference 모델의 per-class classification report 생성 (rebuttal R2).

run_canine_reference.py가 저장한 checkpoint(outputs/canine_reference/canine_s_dual.pt)를
로드해서 test split에 대한 class_report_nat / class_report_lang CSV를 생성한다.
학습 없음 — inference only.

Reviewer HUuH 대응: isolation–F1 상관을 7개 아키텍처 전부로 확장하려면
CANINE의 per-country F1이 필요한데, R1 당시에는 aggregate 지표만 저장했다.

split 재현 조건: run_canine_reference.py와 동일하게
set_seed(config seed) 직후 robust_split을 호출해야 동일한 test set이 나온다.

실행:
  python run_canine_class_report.py --config configs/base_config.yaml
"""

import argparse
import os

import pandas as pd
import torch
import yaml
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader

from data.data_loader import (
    get_loaded_df, get_country_idx, get_country_list, get_lang_group_idx,
    robust_split,
)
from utils.util import set_seed, replace_disallowed_chars
from run_canine_reference import (
    NameDataset, CanineDualHead, MODEL_NAME, OUTPUT_DIR,
    TOKENIZER_EXTRA_TOKENS,
)

EVAL_BATCH_SIZE = 128


def build_test_dataset(config, tokenizer):
    """run_canine_reference.build_splits와 동일 절차 — 단, test split만 토큰화."""
    df = get_loaded_df(config).copy()
    df["Name"] = df["Name"].apply(replace_disallowed_chars)
    df = df[df["Name"].str.len() > 0].reset_index(drop=True)

    country_idx = get_country_idx(config)
    lang_group_idx = get_lang_group_idx(config)
    df["nat_idx"] = df["NOC"].map(country_idx)
    df["lang_idx"] = df["LanguageGroup"].map(lang_group_idx)

    max_length = config["data"]["max_length"] + TOKENIZER_EXTRA_TOKENS

    # 시드 고정 직후 robust_split → run_canine_reference와 동일 분할
    _, _, test_idx = robust_split(df)
    test_df = df.iloc[test_idx]

    return NameDataset(
        test_df["Name"], test_df["nat_idx"], test_df["lang_idx"],
        tokenizer, max_length,
    ), lang_group_idx


@torch.no_grad()
def collect_predictions(model, loader, device):
    model.eval()
    nat_preds, nat_targets, lang_preds, lang_targets = [], [], [], []
    for input_ids, attention_mask, nat_t, lang_t in loader:
        nat_logits, lang_logits = model(
            input_ids.to(device), attention_mask.to(device)
        )
        nat_preds.extend(nat_logits.argmax(dim=1).cpu().tolist())
        lang_preds.extend(lang_logits.argmax(dim=1).cpu().tolist())
        nat_targets.extend(nat_t.tolist())
        lang_targets.extend(lang_t.tolist())
    return nat_preds, nat_targets, lang_preds, lang_targets


def save_class_report(y_true, y_pred, class_names, out_path):
    """Trainer가 저장하는 class_report CSV와 동일 포맷 (행=클래스명 + footer)."""
    report = classification_report(
        y_true, y_pred,
        labels=list(range(len(class_names))),
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    pd.DataFrame(report).T.to_csv(out_path)
    print(f"저장: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=os.path.join("configs", "base_config.yaml"))
    parser.add_argument("--checkpoint",
                        default=os.path.join(OUTPUT_DIR, "canine_s_dual.pt"))
    args = parser.parse_args()

    from transformers import CanineTokenizer

    with open(args.config, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    set_seed(config["experiment"]["seed"])

    tokenizer = CanineTokenizer.from_pretrained(MODEL_NAME)
    test_ds, lang_group_idx = build_test_dataset(config, tokenizer)
    print(f"[Data] test={len(test_ds)}")

    model = CanineDualHead(
        MODEL_NAME,
        n_classes=config["model"]["num_classes"],
        n_lang_groups=config["model"]["num_lang_groups"],
    ).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))

    loader = DataLoader(test_ds, batch_size=EVAL_BATCH_SIZE, num_workers=2,
                        pin_memory=True)
    nat_preds, nat_targets, lang_preds, lang_targets = collect_predictions(
        model, loader, device,
    )

    # 정합성 검사: R1에서 보고한 Nat-T1 48.63과 일치해야 split 재현이 맞음
    nat_top1 = 100.0 * sum(p == t for p, t in zip(nat_preds, nat_targets)) / len(nat_targets)
    print(f"[Sanity] Nat-T1 = {nat_top1:.2f} (R1 보고값 48.63)")

    country_list = get_country_list(config)
    lang_list = [g for g, _ in sorted(lang_group_idx.items(), key=lambda kv: kv[1])]

    save_class_report(
        nat_targets, nat_preds, country_list,
        os.path.join(OUTPUT_DIR, "class_report_nat_canine_s.csv"),
    )
    save_class_report(
        lang_targets, lang_preds, lang_list,
        os.path.join(OUTPUT_DIR, "class_report_lang_canine_s.csv"),
    )


if __name__ == "__main__":
    main()
