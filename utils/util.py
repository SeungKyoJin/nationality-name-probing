import torch
import pandas as pd
import numpy as np
import random
import os


alphabets = 'abcdefghijklmnopqrstuvwxyz '
alphabets = list(alphabets)
alphabets.insert(0, '[')
alphabets.append(']')

char_to_idx = {char: idx for idx, char in enumerate(alphabets)}
trigram_dict = {}
trigram_idx = 0
for first_c in alphabets[:-1]:
    for second_c in alphabets:
        if second_c == '[' or second_c == ']':
            continue
        for third_c in alphabets[1:]:
            trigram_dict[''.join([first_c, second_c, third_c])] = trigram_idx
            trigram_idx += 1

allowed_chars = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't',
                     'u', 'v', 'w', 'x', 'y', 'z', ' ']


def replace_disallowed_chars(text):
    return ''.join([char.lower() if char.lower() in allowed_chars else '' for char in text])


def generate_unigrams(text):
    unigrams = [text[i] for i in range(len(text) - 1)]
    return unigrams


def generate_bigrams(text):
    bigrams = [text[i:i+2] for i in range(len(text) - 1)]
    return bigrams


def generate_trigrams(text):
    trigrams = [text[i:i+3] for i in range(len(text) - 2)]
    return trigrams


def text_to_bigram_indices(text, bigram_to_idx, max_length, padding_token=0):
    bigrams = generate_bigrams(text)
    indices = [bigram_to_idx.get(bigram, padding_token) for bigram in bigrams]  # bigram 인덱스 리스트 생성
    if len(indices) < max_length:
        indices.extend([padding_token] * (max_length - len(indices)))  # 패딩 추가
    else:
        indices = indices[:max_length]
    return indices


def text_to_trigram_indices(text, trigram_to_idx, max_length, padding_token=0):
    trigrams = generate_trigrams(text)
    indices = [trigram_to_idx.get(trigram, padding_token) for trigram in trigrams]  # trigram 인덱스 리스트 생성
    if len(indices) < max_length:
        indices.extend([padding_token] * (max_length - len(indices)))  # 패딩 추가
    else:
        indices = indices[:max_length]
    return indices


def text_to_unigram_indices(text, char_to_idx, max_length, padding_token=0):
    indices = [char_to_idx.get(char, padding_token) for char in text.lower()]  # 알파벳 외는 0
    if len(indices) < max_length:
        indices.extend([padding_token] * (max_length - len(indices)))  # 패딩 추가
    else:
        indices = indices[:max_length]
    return indices


def text_to_mixed_indices(text, mixed_to_idx, max_length, padding_token=0):
    unigrams = generate_unigrams(text)
    bigrams = generate_bigrams(text)
    trigrams = generate_trigrams(text)

    mixed_ngrams = []
    mixed_ngrams.extend(unigrams)
    mixed_ngrams.extend(bigrams)
    mixed_ngrams.extend(trigrams)

    indices = [mixed_to_idx.get(ngram, padding_token) for ngram in mixed_ngrams]  # trigram 인덱스 리스트 생성
    if len(indices) < max_length:
        indices.extend([padding_token] * (max_length - len(indices)))  # 패딩 추가
    else:
        indices = indices[:max_length]
    return indices


def get_ngram_dict(ngram):
    ngram_dict = {}
    ngram_dict['<PAD>'] = 0
    ngram_dict['<UNK>'] = 1
    ngram_idx = 2
    if ngram == 'uni':
        for c in alphabets:
            ngram_dict[c] = ngram_idx
            ngram_idx += 1
    elif ngram == 'bi':
        for first_c in alphabets:
            if first_c == ']':
                continue
            for second_c in alphabets:
                if second_c == '[':
                    continue
                ngram_dict[''.join([first_c, second_c])] = ngram_idx
                ngram_idx += 1
    elif ngram == 'tri':
        for first_c in alphabets:
            if first_c == ']':
                continue
            for second_c in alphabets:
                if second_c == '[' or second_c == ']':
                    continue
                for third_c in alphabets:
                    if third_c == '[':
                        continue
                    ngram_dict[''.join([first_c, second_c, third_c])] = ngram_idx
                    ngram_idx += 1
    elif ngram == 'mixed' or ngram == 'multi':
        for c in alphabets: # uni
            ngram_dict[c] = ngram_idx
            ngram_idx += 1
        for first_c in alphabets: # bi
            if first_c == ']':
                continue
            for second_c in alphabets:
                if second_c == '[':
                    continue
                ngram_dict[''.join([first_c, second_c])] = ngram_idx
                ngram_idx += 1
        for first_c in alphabets: # tri
            if first_c == ']':
                continue
            for second_c in alphabets:
                if second_c == '[' or second_c == ']':
                    continue
                for third_c in alphabets:
                    if third_c == '[':
                        continue
                    ngram_dict[''.join([first_c, second_c, third_c])] = ngram_idx
                    ngram_idx += 1

    return ngram_dict


def set_seed(seed: int = 42):
    """
    재현성을 위해 모든 라이브러리의 시드를 고정합니다.
    """
    # 1. 파이썬 기본 random 시드 고정
    random.seed(seed)

    # 2. 넘파이(NumPy) 시드 고정 (데이터 전처리 등에 사용)
    np.random.seed(seed)

    # 3. 파이토치(PyTorch) CPU 시드 고정
    torch.manual_seed(seed)

    # 4. 파이토치(PyTorch) GPU 시드 고정 (단일/멀티 GPU 대응)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # 5. CuDNN 백엔드 고정 (속도가 약간 느려질 수 있지만 재현성 확보에 필수)
    # True로 설정하면 연산 결과가 항상 동일하도록 강제합니다.
    torch.backends.cudnn.deterministic = True

    # 연산 최적화 알고리즘을 고정하여 동일한 연산 결과를 보장합니다.
    torch.backends.cudnn.benchmark = False

    # 6. 환경 변수 고정 (Hash 기반 무작위성 방지)
    os.environ['PYTHONHASHSEED'] = str(seed)

    print(f"[SEED SET] All seeds have been set to: {seed}")


def merge_config(config, args):
    """CLI 인자로 YAML config 값 override."""
    mapping = {
        'ngram': ('data', 'ngram'),
        'max_length': ('data', 'max_length'),
        'model_type': ('model', 'type'),
        'embedding_dim': ('model', 'embedding_dim'),
        'hidden_dim': ('model', 'hidden_dim'),
        'num_layers': ('model', 'n_layers'),
        'dropout': ('model', 'dropout'),
        'epochs': ('train', 'epochs'),
        'batch_size': ('train', 'batch_size'),
        'lr': ('train', 'learning_rate'),
        'weight_decay': ('train', 'weight_decay'),
        'alpha': ('train', 'alpha'),
        'save_dir': ('train', 'save_dir'),
        'save_name': ('train', 'save_name'),
    }

    for arg_name, (section, key) in mapping.items():
        val = getattr(args, arg_name, None)
        if val is not None:
            config[section][key] = val

    return config


def save_result(config, metrics):
    """학습 완료 후 주요 하이퍼파라미터와 5개 지표를 CSV에 누적 저장."""
    result_file = os.path.join(
        config['experiment']['result_directory'],
        config['train']['save_dir'],
        'experiment_results.csv',
    )

    new_result = {
        "save_name": config['train']['save_name'],
        "ngram": config['data']['ngram'],
        "lr": config['train']['learning_rate'],
        "batch_size": config['train']['batch_size'],
        "max_length": config['data']['max_length'],
        "weight_decay": config['train']['weight_decay'],
        "alpha": config['train'].get('alpha', None),
        # Architecture options (신규)
        "model_type": config['model'].get('type'),
        "fusion_type": config['model'].get('fusion_type', 'linear'),
        "pool_type": config['model'].get('pool_type', 'attention'),
        # Contrastive (신규)
        "contrastive_weight": config['train'].get('contrastive_weight', 0.0),
        "contrastive_variant": config['train'].get('contrastive_variant', None),
        # 국적 지표
        "nat_top1": metrics.get('nat_top1'),
        "nat_top5": metrics.get('nat_top5'),
        "nat_macro_f1": metrics.get('nat_macro_f1'),
        "nat_weighted_f1": metrics.get('nat_weighted_f1'),
        # 언어권 지표
        "lang_top1": metrics.get('lang_top1'),
        "lang_top3": metrics.get('lang_top3'),
        "lang_top5": metrics.get('lang_top5'),
        "lang_macro_f1": metrics.get('lang_macro_f1'),
        "lang_weighted_f1": metrics.get('lang_weighted_f1'),
        # Phase 4 — CNL 상관관계 (신규, 자동 기록)
        "cnl_rho_all": metrics.get('cnl_rho_all'),
        "cnl_rho_subset": metrics.get('cnl_rho_subset'),
        "cnl_n_pairs": metrics.get('cnl_n_pairs'),
        "test_loss": metrics.get('loss'),
        "timestamp": pd.Timestamp.now(),
    }
    df = pd.DataFrame([new_result])

    if not os.path.exists(result_file):
        df.to_csv(result_file, index=False)
    else:
        df.to_csv(result_file, mode='a', header=False, index=False)
