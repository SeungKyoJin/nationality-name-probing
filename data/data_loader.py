from .dataset import TextDataset, MultiChannelDataset
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from utils import util
from utils.augmentation import augment_minority_classes
from functools import lru_cache

import os
import numpy as np
import pandas as pd
import torch


REQUIRED_COLUMNS = ('Name', 'NOC', 'LanguageGroup')


# --------------------------------------------------------------------------
# CSV 로딩 (lru_cache로 중복 I/O 제거)
# --------------------------------------------------------------------------

@lru_cache(maxsize=4)
def _load_raw_df(raw_path: str, dataset_filter: str = None) -> pd.DataFrame:
    """
    concat_names_mapped.csv 로딩.
    - 필수 컬럼: Name, NOC, LanguageGroup
    - 선택 컬럼: dataset (ablation용 필터링 가능)
    - 결측 행 제거
    """
    df = pd.read_csv(raw_path)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"필수 컬럼 누락: {missing} / 실제 컬럼: {list(df.columns)}")

    df = df.dropna(subset=list(REQUIRED_COLUMNS)).copy()

    if dataset_filter is not None:
        if 'dataset' not in df.columns:
            raise ValueError("dataset_filter 지정됨. 하지만 CSV에 'dataset' 컬럼이 없음")
        df = df[df['dataset'] == dataset_filter].copy()

    return df.reset_index(drop=True)


def get_loaded_df(config) -> pd.DataFrame:
    return _load_raw_df(
        config['data']['raw_path'],
        config['data'].get('dataset_filter'),
    )


# --------------------------------------------------------------------------
# 라벨 매핑 (NOC + LanguageGroup)
# --------------------------------------------------------------------------

def get_country_list(config):
    """NOC 사전순 정렬 리스트."""
    return sorted(get_loaded_df(config)['NOC'].unique().tolist())


def get_country_idx(config):
    return {noc: i for i, noc in enumerate(get_country_list(config))}


def get_lang_group_list(config):
    """LanguageGroup 사전순 정렬 리스트."""
    return sorted(get_loaded_df(config)['LanguageGroup'].unique().tolist())


def get_lang_group_idx(config):
    return {lg: i for i, lg in enumerate(get_lang_group_list(config))}


def get_num_labels(config):
    """국적(NOC) 라벨 인덱스 리스트."""
    return list(range(config['model']['num_classes']))


def get_num_lang_labels(config):
    """언어권 라벨 인덱스 리스트."""
    return list(range(config['model']['num_lang_groups']))


# --------------------------------------------------------------------------
# Vocab
# --------------------------------------------------------------------------

def get_vocab_size(config):
    ngram = config['data']['ngram']
    if ngram in ('uni', 'bi', 'tri', 'mixed'):
        return len(util.get_ngram_dict(ngram))
    elif ngram == 'multi':
        return {t: len(util.get_ngram_dict(t)) for t in ('uni', 'bi', 'tri')}
    raise ValueError(f"지원하지 않는 ngram 방식입니다: {ngram}")


def get_ngram_dicts(ngram):
    if ngram in ('uni', 'bi', 'tri', 'mixed'):
        return util.get_ngram_dict(ngram)
    elif ngram == 'multi':
        return {t: util.get_ngram_dict(t) for t in ('uni', 'bi', 'tri')}
    raise ValueError(f"지원하지 않는 ngram 방식입니다: {ngram}")


# --------------------------------------------------------------------------
# 데이터 분할 (NOC 기준 stratified split)
# --------------------------------------------------------------------------

def robust_split(df, label_col='NOC', train_size=0.8):
    """
    NOC 기준 stratified split.
    - n>=3: 8:1:1
    - n==2: train / test
    - n==1: train만
    """
    train_indices, val_indices, test_indices = [], [], []

    for _, group in df.groupby(label_col):
        idxs = group.index.values.copy()
        np.random.shuffle(idxs)
        n = len(idxs)

        if n >= 3:
            t_idx, temp_idx = train_test_split(idxs, train_size=train_size, random_state=42)
            v_idx, te_idx = train_test_split(temp_idx, test_size=0.5, random_state=42)
            train_indices.extend(t_idx)
            val_indices.extend(v_idx)
            test_indices.extend(te_idx)
        elif n == 2:
            train_indices.append(idxs[0])
            test_indices.append(idxs[1])
        else:
            train_indices.append(idxs[0])

    return train_indices, val_indices, test_indices


def get_datasets(dataset_obj, df):
    from torch.utils.data import Subset
    indices = robust_split(df)
    return [Subset(dataset_obj, idx) for idx in indices]


# --------------------------------------------------------------------------
# Class weights 계산
# --------------------------------------------------------------------------

def compute_class_weights(labels: pd.Series, label_to_idx: dict) -> torch.FloatTensor:
    """
    Inverse frequency class weights.

    weight_i = total_samples / (num_classes * count_i)

    빈도가 낮은 클래스일수록 높은 가중치를 받아서
    CrossEntropyLoss가 소수 클래스의 오류를 더 크게 반영한다.
    """
    counts = labels.value_counts()
    n_classes = len(label_to_idx)
    total = len(labels)

    weights = torch.ones(n_classes)
    for label, idx in label_to_idx.items():
        if label in counts.index:
            weights[idx] = total / (n_classes * counts[label])

    return weights


# --------------------------------------------------------------------------
# Dataset 생성 헬퍼
# --------------------------------------------------------------------------

def _make_dataset(sub_df, ngram, max_length, ngram_dicts, country_idx, lang_group_idx):
    """DataFrame에서 Dataset 객체를 생성한다."""
    if ngram == 'multi':
        return MultiChannelDataset(
            texts=sub_df['Name'],
            labels=sub_df['NOC'],
            lang_labels=sub_df['LanguageGroup'],
            uni_ngram_dict=ngram_dicts['uni'],
            bi_ngram_dict=ngram_dicts['bi'],
            tri_ngram_dict=ngram_dicts['tri'],
            max_length=max_length,
            country_idx=country_idx,
            lang_group_idx=lang_group_idx,
        )
    else:
        return TextDataset(
            texts=sub_df['Name'],
            char_to_idx=ngram_dicts,
            max_length=max_length,
            country_idx=country_idx,
            lang_group_idx=lang_group_idx,
            n_gram=ngram,
            labels=sub_df['NOC'],
            lang_labels=sub_df['LanguageGroup'],
        )


# --------------------------------------------------------------------------
# 메인 파이프라인
# --------------------------------------------------------------------------

def prepare_data(config):
    """
    데이터 로딩 → 전처리 → 분할 → (train만) augmentation → class weights 계산.

    Returns:
        train_loader, val_loader, test_loader, nat_weights, lang_weights
        - nat_weights: 국적 inverse frequency weights (Tensor, shape=[num_classes])
        - lang_weights: 언어권 inverse frequency weights (Tensor, shape=[num_lang_groups])
        - class_weight 비활성화 시 nat_weights, lang_weights = None, None
    """
    ngram = config['data']['ngram']
    max_length = config['data']['max_length']
    batch_size = config['train']['batch_size']

    # 1. 로딩 & 전처리
    df = get_loaded_df(config).copy()
    df['Name'] = df['Name'].apply(util.replace_disallowed_chars)
    df = df[df['Name'].str.len() > 0].reset_index(drop=True)

    country_idx = get_country_idx(config)
    lang_group_idx = get_lang_group_idx(config)
    ngram_dicts = get_ngram_dicts(ngram)

    # 2. train/val/test 분할 (원본 데이터만으로 분할)
    train_indices, val_indices, test_indices = robust_split(df)
    train_df = df.iloc[train_indices].reset_index(drop=True)
    val_df = df.iloc[val_indices].reset_index(drop=True)
    test_df = df.iloc[test_indices].reset_index(drop=True)

    # 3. LLM 합성 데이터 추가 (train만)
    synthetic_path = config['data'].get('synthetic_path')
    if synthetic_path and os.path.exists(synthetic_path):
        syn_target = int(config['data'].get('synthetic_target', 0))
        syn_df = pd.read_csv(synthetic_path)
        syn_df = syn_df[syn_df['NOC'].isin(country_idx.keys())].copy()
        syn_df['Name'] = syn_df['Name'].apply(util.replace_disallowed_chars)
        syn_df = syn_df[syn_df['Name'].str.len() > 0].reset_index(drop=True)
        syn_df = syn_df[['Name', 'NOC', 'LanguageGroup']].copy()

        if syn_target > 0:
            # 클래스별 train 원본 수 기준으로 부족분만 추가
            train_counts = train_df['NOC'].value_counts()
            filtered = []
            for noc, group in syn_df.groupby('NOC'):
                current = train_counts.get(noc, 0)
                if current < syn_target:
                    needed = syn_target - current
                    filtered.append(group.head(needed))
            if filtered:
                syn_df = pd.concat(filtered, ignore_index=True)
            else:
                syn_df = syn_df.head(0)

        train_df = pd.concat([train_df, syn_df], ignore_index=True)
        print(f"[Synthetic] {len(syn_df)}건 추가 (target={syn_target}) → train 총 {len(train_df)}건")

    # 4. 소수 클래스 char augmentation (train만)
    if config['data'].get('augment', False):
        target = int(config['data'].get('augment_target', 200))
        train_df = augment_minority_classes(train_df, target_count=target)

    # 4. Class weights 계산 (augmentation 이후 분포 기준)
    nat_weights = None
    lang_weights = None
    if config['train'].get('class_weight', False):
        nat_weights = compute_class_weights(train_df['NOC'], country_idx)
        lang_weights = compute_class_weights(train_df['LanguageGroup'], lang_group_idx)
        print(
            f"[Class Weights] nat: min={nat_weights.min():.3f} max={nat_weights.max():.3f} | "
            f"lang: min={lang_weights.min():.3f} max={lang_weights.max():.3f}"
        )

    # 5. Dataset & DataLoader 생성
    common = dict(
        ngram=ngram, max_length=max_length,
        ngram_dicts=ngram_dicts, country_idx=country_idx,
        lang_group_idx=lang_group_idx,
    )
    train_ds = _make_dataset(train_df, **common)
    val_ds = _make_dataset(val_df, **common)
    test_ds = _make_dataset(test_df, **common)

    # Speed: num_workers (parallel data loading) + pin_memory (faster GPU transfer)
    _dl_kwargs = dict(num_workers=4, pin_memory=True, persistent_workers=True)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, **_dl_kwargs)
    val_loader = DataLoader(val_ds, batch_size=batch_size, **_dl_kwargs)
    test_loader = DataLoader(test_ds, batch_size=batch_size, **_dl_kwargs)

    return train_loader, val_loader, test_loader, nat_weights, lang_weights
