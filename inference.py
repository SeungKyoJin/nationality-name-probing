import os
import time
import argparse

import torch
import yaml
import pandas as pd
from sklearn.metrics import classification_report, f1_score

from data import (
    get_country_list, get_lang_group_list,
    get_ngram_dicts, get_vocab_size,
    get_num_labels, get_num_lang_labels,
)
from models import get_model
from utils import util


class NameInference:
    """
    외부 CSV를 읽어 국적(NOC) + 언어권(LanguageGroup) 두 태스크 동시 평가.
    - CSV 필수 컬럼: Name, NOC
    - CSV 선택 컬럼: LanguageGroup (있으면 언어권 지표도 계산)
    """

    _SINGLE_CONVERTERS = {
        'uni': util.text_to_unigram_indices,
        'bi': util.text_to_bigram_indices,
        'tri': util.text_to_trigram_indices,
        'mixed': util.text_to_mixed_indices,
    }

    def __init__(self, model, config):
        self.config = config
        self.device = config['experiment']['device']
        self.model = model.to(self.device)
        self.model.eval()

        self.country_list = get_country_list(config)
        self.country_to_idx = {noc: i for i, noc in enumerate(self.country_list)}

        self.lang_list = get_lang_group_list(config)
        self.lang_to_idx = {lg: i for i, lg in enumerate(self.lang_list)}

        self.num_labels = get_num_labels(config)
        self.num_lang_labels = get_num_lang_labels(config)

        self.max_len = config['data']['max_length']
        self.ngram = config['data']['ngram']
        self.ngram_dicts = get_ngram_dicts(self.ngram)

    # ------------------------------------------------------------------
    # 전처리
    # ------------------------------------------------------------------
    def preprocess(self, name: str):
        """ngram 모드에 따라 모델 입력 튜플 반환."""
        name = util.replace_disallowed_chars(str(name).lower())

        if self.ngram == 'multi':
            uni = util.text_to_unigram_indices(name, self.ngram_dicts['uni'], self.max_len)
            bi = util.text_to_bigram_indices(name, self.ngram_dicts['bi'], self.max_len)
            tri = util.text_to_trigram_indices(name, self.ngram_dicts['tri'], self.max_len)
            return (
                torch.LongTensor(uni).unsqueeze(0).to(self.device),
                torch.LongTensor(bi).unsqueeze(0).to(self.device),
                torch.LongTensor(tri).unsqueeze(0).to(self.device),
            )

        if self.ngram in self._SINGLE_CONVERTERS:
            indices = self._SINGLE_CONVERTERS[self.ngram](
                name, self.ngram_dicts, self.max_len
            )
            return (torch.LongTensor(indices).unsqueeze(0).to(self.device),)

        raise ValueError(f"지원하지 않는 ngram 방식입니다: {self.ngram}")

    # ------------------------------------------------------------------
    # 외부 CSV 평가
    # ------------------------------------------------------------------
    def evaluate_csv(self, csv_path: str, save_name: str = "external_test"):
        df = pd.read_csv(csv_path)

        if 'Name' not in df.columns or 'NOC' not in df.columns:
            raise ValueError("CSV에는 최소 'Name', 'NOC' 컬럼이 있어야 합니다.")

        has_lang = 'LanguageGroup' in df.columns

        y_true_nat, y_pred_nat = [], []
        y_true_lang, y_pred_lang = [], []
        correct_nat = correct_top5 = correct_lang = 0
        total = skipped = 0

        with torch.no_grad():
            for _, row in df.iterrows():
                name = row['Name']
                noc = row['NOC']

                # 학습에 없는 NOC는 공정성을 위해 제외
                if noc not in self.country_to_idx:
                    skipped += 1
                    continue

                inputs = self.preprocess(name)
                nat_logits, lang_logits = self.model(*inputs)

                # Nationality
                _, top5_idx = torch.topk(nat_logits, k=5, dim=1)
                nat_pred = top5_idx[0, 0].item()
                target_idx = self.country_to_idx[noc]

                total += 1
                y_true_nat.append(target_idx)
                y_pred_nat.append(nat_pred)
                if nat_pred == target_idx:
                    correct_nat += 1
                if target_idx in top5_idx[0].tolist():
                    correct_top5 += 1

                # Language group (선택)
                if has_lang:
                    lg = row['LanguageGroup']
                    if lg in self.lang_to_idx:
                        lang_pred = lang_logits.argmax(dim=1).item()
                        lang_target = self.lang_to_idx[lg]
                        y_true_lang.append(lang_target)
                        y_pred_lang.append(lang_pred)
                        if lang_pred == lang_target:
                            correct_lang += 1

        if total == 0:
            raise ValueError("평가 가능한 샘플이 0개입니다. 학습 NOC와 겹치는지 확인하세요.")

        # ---- 지표 ----
        metrics = {
            'nat_top1': correct_nat / total * 100,
            'nat_top5': correct_top5 / total * 100,
            'nat_macro_f1': f1_score(
                y_true_nat, y_pred_nat,
                labels=self.num_labels, average='macro', zero_division=0,
            ) * 100,
            'lang_top1': None,
            'lang_macro_f1': None,
            'total': total,
            'skipped': skipped,
        }

        if y_true_lang:
            metrics['lang_top1'] = correct_lang / len(y_true_lang) * 100
            metrics['lang_macro_f1'] = f1_score(
                y_true_lang, y_pred_lang,
                labels=self.num_lang_labels, average='macro', zero_division=0,
            ) * 100

        self._print_metrics(metrics)
        self._save_reports(y_true_nat, y_pred_nat, y_true_lang, y_pred_lang, save_name)

        return metrics

    # ------------------------------------------------------------------
    # 헬퍼
    # ------------------------------------------------------------------
    def _print_metrics(self, m):
        print(f"\n[External Eval] evaluated={m['total']} skipped={m['skipped']}")
        print(
            f"  Nationality: Top-1={m['nat_top1']:.2f}% | "
            f"Top-5={m['nat_top5']:.2f}% | Macro-F1={m['nat_macro_f1']:.2f}%"
        )
        if m['lang_top1'] is not None:
            print(
                f"  Language:    Top-1={m['lang_top1']:.2f}% | "
                f"Macro-F1={m['lang_macro_f1']:.2f}%"
            )
        else:
            print("  Language:    (LanguageGroup 컬럼 없음 — 평가 생략)")

    def _save_reports(self, y_true_nat, y_pred_nat, y_true_lang, y_pred_lang, save_name):
        save_dir = self.config['experiment']['result_directory']
        os.makedirs(save_dir, exist_ok=True)
        prefix = self.config['inference'].get('report_prefix', 'external')

        # Nationality report
        present_nat = sorted(set(y_true_nat))
        nat_report = classification_report(
            y_true_nat, y_pred_nat,
            labels=present_nat,
            target_names=[self.country_list[i] for i in present_nat],
            output_dict=True, zero_division=0,
        )
        nat_path = os.path.join(save_dir, f"{prefix}_nat_{save_name}.csv")
        pd.DataFrame(nat_report).transpose().to_csv(nat_path, index=True)
        print(f"[Report] 국적: {nat_path}")

        # Language group report (있으면)
        if y_true_lang:
            present_lang = sorted(set(y_true_lang))
            lang_report = classification_report(
                y_true_lang, y_pred_lang,
                labels=present_lang,
                target_names=[self.lang_list[i] for i in present_lang],
                output_dict=True, zero_division=0,
            )
            lang_path = os.path.join(save_dir, f"{prefix}_lang_{save_name}.csv")
            pd.DataFrame(lang_report).transpose().to_csv(lang_path, index=True)
            print(f"[Report] 언어권: {lang_path}")


# ----------------------------------------------------------------------
# 엔트리포인트
# ----------------------------------------------------------------------

def run_inference(config):
    vocab_size = get_vocab_size(config)
    model = get_model(
        config, vocab_size,
        n_classes=config['model']['num_classes'],
        n_lang_groups=config['model']['num_lang_groups'],
    )

    model_path = config['inference']['model_path']
    state_dict = torch.load(model_path, map_location=config['experiment']['device'])
    model.load_state_dict(state_dict)

    infer = NameInference(model, config)
    return infer.evaluate_csv(config['inference']['test_dataset_path'])


if __name__ == "__main__":
    start = time.time()

    parser = argparse.ArgumentParser(description="Nationality Classification Inference")
    parser.add_argument("--config", type=str, default=os.path.join("configs", "base_config.yaml"))
    args = parser.parse_args()

    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    metrics = run_inference(config)
    print(f"\n추론 시간: {time.time() - start:.2f}s")
