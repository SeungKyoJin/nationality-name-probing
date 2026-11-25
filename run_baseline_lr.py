"""
Logistic Regression Baseline.

sklearn 기반이라 PyTorch Trainer와 별도로 실행.
Character n-gram TF-IDF feature → LR → nationality / language group 각각 분류.

Usage:
    python run_baseline_lr.py --config configs/base_config.yaml
"""

import os
import argparse
import time
from datetime import datetime

import yaml
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report, top_k_accuracy_score
)

from utils.util import set_seed, replace_disallowed_chars


def load_and_split(config):
    """데이터 로딩 + 전처리 + stratified split."""
    df = pd.read_csv(config['data']['raw_path'])
    df = df.dropna(subset=['Name', 'NOC', 'LanguageGroup']).copy()

    dataset_filter = config['data'].get('dataset_filter')
    if dataset_filter:
        df = df[df['dataset'] == dataset_filter].copy()

    df['Name'] = df['Name'].apply(replace_disallowed_chars)
    df = df[df['Name'].str.len() > 0].reset_index(drop=True)

    # Stratified split (data_loader.robust_split과 동일 로직)
    from sklearn.model_selection import train_test_split
    train_idx, val_idx, test_idx = [], [], []

    for _, group in df.groupby('NOC'):
        idxs = group.index.values.copy()
        np.random.shuffle(idxs)
        n = len(idxs)
        if n >= 3:
            t, temp = train_test_split(idxs, train_size=0.8, random_state=42)
            v, te = train_test_split(temp, test_size=0.5, random_state=42)
            train_idx.extend(t)
            val_idx.extend(v)
            test_idx.extend(te)
        elif n == 2:
            train_idx.append(idxs[0])
            test_idx.append(idxs[1])
        else:
            train_idx.append(idxs[0])

    train_df = df.iloc[train_idx].reset_index(drop=True)
    val_df = df.iloc[val_idx].reset_index(drop=True)
    test_df = df.iloc[test_idx].reset_index(drop=True)

    return train_df, val_df, test_df


def evaluate(model, X, y_true, label_list, task_name, top_k_list=(3, 5)):
    """분류 결과 평가 — 딥러닝 trainer.evaluate와 동일 지표 5종.

    반환: top1, top3, top5, macro_f1, weighted_f1, report
    """
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)

    top1 = accuracy_score(y_true, y_pred) * 100

    # Top-3 / Top-5 accuracy
    topk = {}
    for k in top_k_list:
        if y_proba.shape[1] >= k:
            topk[k] = top_k_accuracy_score(y_true, y_proba, k=k) * 100
        else:
            topk[k] = None

    macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0) * 100
    weighted_f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0) * 100

    print(f"\n[{task_name}]")
    print(f"  Top-1 Accuracy:  {top1:.2f}%")
    for k in top_k_list:
        if topk.get(k) is not None:
            print(f"  Top-{k} Accuracy:  {topk[k]:.2f}%")
    print(f"  Macro-F1:        {macro_f1:.2f}%")
    print(f"  Weighted-F1:     {weighted_f1:.2f}%")

    report = classification_report(
        y_true, y_pred, target_names=label_list,
        output_dict=True, zero_division=0,
    )

    return {
        'top1': top1,
        'top3': topk.get(3),
        'top5': topk.get(5),
        'macro_f1': macro_f1,
        'weighted_f1': weighted_f1,
        'report': report,
    }


def main():
    parser = argparse.ArgumentParser(description="LR Baseline")
    parser.add_argument("--config", type=str, default=os.path.join("configs", "base_config.yaml"))
    args = parser.parse_args()

    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    set_seed(config['experiment']['seed'])
    start = time.time()

    print("=" * 60)
    print(" Logistic Regression Baseline")
    print("=" * 60)

    # 1. 데이터 로딩
    print("\n[1/4] 데이터 로딩 및 분할")
    train_df, val_df, test_df = load_and_split(config)
    print(f"  Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

    # 2. TF-IDF feature 추출 (unigram + bigram + trigram character n-gram)
    print("\n[2/4] TF-IDF feature 추출 (char n-gram 1~3)")
    vectorizer = TfidfVectorizer(
        analyzer='char',
        ngram_range=(1, 3),
        max_features=50000,
        lowercase=True,
    )
    X_train = vectorizer.fit_transform(train_df['Name'])
    X_val = vectorizer.transform(val_df['Name'])
    X_test = vectorizer.transform(test_df['Name'])
    print(f"  Feature 차원: {X_train.shape[1]}")

    # 라벨
    country_list = sorted(train_df['NOC'].unique())
    lang_list = sorted(train_df['LanguageGroup'].unique())

    # 3. Nationality LR
    # NOTE: sklearn ≥1.7 에서 multi_class 인자 제거됨 (auto-detect).
    print("\n[3/4] Nationality LR 학습")
    lr_nat = LogisticRegression(
        max_iter=1000, C=1.0, solver='saga',
        n_jobs=-1, verbose=1,
    )
    lr_nat.fit(X_train, train_df['NOC'])

    nat_results = evaluate(
        lr_nat, X_test, test_df['NOC'],
        country_list, "Nationality",
    )

    # 4. Language Group LR
    print("\n[4/4] Language Group LR 학습")
    lr_lang = LogisticRegression(
        max_iter=1000, C=1.0, solver='saga',
        n_jobs=-1, verbose=1,
    )
    lr_lang.fit(X_train, train_df['LanguageGroup'])

    lang_results = evaluate(
        lr_lang, X_test, test_df['LanguageGroup'],
        lang_list, "Language Group",
    )

    # 결과 저장 — 딥러닝 실험과 동일 디렉토리 구조 (outputs/checkpoints/<date>/)
    date_dir = datetime.now().strftime("%y%m%d")
    result_dir = os.path.join(
        config['experiment']['result_directory'],   # outputs
        config['train']['save_dir'],                # checkpoints
        date_dir,                                    # 260506
    )
    os.makedirs(result_dir, exist_ok=True)

    # CNL ρ 자동 계산 — 딥러닝 train.py의 compute_cnl_rho_quick과 동일 패턴.
    # LR은 임베딩이 없으므로 nationality LR의 coef_ vector를 country representation으로
    # 사용해서 pairwise cosine → DICL CNL과 Spearman ρ.
    cnl_metrics = {}
    try:
        from analysis.dicl_lr import compute_cnl_rho_lr
        cnl_metrics = compute_cnl_rho_lr(lr_nat) or {}
        if cnl_metrics:
            print(f"[CNL auto-log] ρ_all={cnl_metrics['cnl_rho_all']:.4f}  "
                  f"ρ_subset={cnl_metrics['cnl_rho_subset']:.4f}  "
                  f"n={cnl_metrics['cnl_n_pairs']}")
    except Exception as e:
        print(f"[WARN] CNL auto-log 실패 (분류 결과는 정상): {e}")

    # 딥러닝 실험 column 순서와 정확히 일치 (cf. trainer.evaluate + train.py 자동로그)
    result = {
        "save_name": "lr_baseline",
        "ngram": "tfidf_char_1-3",
        "lr": "C=1.0",
        "batch_size": "-",
        "max_length": "-",
        "weight_decay": "-",
        "alpha": "-",
        "model_type": "logistic_regression",
        "fusion_type": "-",
        "pool_type": "-",
        "contrastive_weight": 0.0,
        "contrastive_variant": "-",
        "nat_top1":          nat_results['top1'],
        "nat_top5":          nat_results['top5'],
        "nat_macro_f1":      nat_results['macro_f1'],
        "nat_weighted_f1":   nat_results['weighted_f1'],
        "lang_top1":         lang_results['top1'],
        "lang_top3":         lang_results['top3'],
        "lang_top5":         lang_results['top5'],
        "lang_macro_f1":     lang_results['macro_f1'],
        "lang_weighted_f1":  lang_results['weighted_f1'],
        "cnl_rho_all":       cnl_metrics.get('cnl_rho_all', '-'),
        "cnl_rho_subset":    cnl_metrics.get('cnl_rho_subset', '-'),
        "cnl_n_pairs":       cnl_metrics.get('cnl_n_pairs', '-'),
        "test_loss":         "-",
        "timestamp":         pd.Timestamp.now(),
    }

    result_file = os.path.join(result_dir, "experiment_results.csv")
    df_result = pd.DataFrame([result])
    if os.path.exists(result_file):
        df_result.to_csv(result_file, mode='a', header=False, index=False)
    else:
        df_result.to_csv(result_file, index=False)

    # Per-class report 저장
    nat_report_df = pd.DataFrame(nat_results['report']).transpose()
    lang_report_df = pd.DataFrame(lang_results['report']).transpose()
    nat_report_df.to_csv(os.path.join(result_dir, "class_report_nat_lr_baseline.csv"))
    lang_report_df.to_csv(os.path.join(result_dir, "class_report_lang_lr_baseline.csv"))

    print(f"\n{'=' * 60}")
    print(f" LR Baseline 완료 ({time.time() - start:.1f}s)")
    print(f" Nat Top-1: {nat_results['top1']:.2f}% | "
          f"Top-5: {nat_results['top5']:.2f}% | "
          f"F1: {nat_results['macro_f1']:.2f}%")
    print(f" Lang Top-1: {lang_results['top1']:.2f}% | "
          f"F1: {lang_results['macro_f1']:.2f}%")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    main()
