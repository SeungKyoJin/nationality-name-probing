"""
소수 클래스 Character-level Augmentation 스크립트.

원본 CSV를 읽어 train/val/test 분할 후, training set의 소수 클래스를
target_count까지 augmentation으로 채운 뒤 별도 CSV로 저장한다.

Usage:
    python augment_data.py
"""

import csv
import random
import os
from collections import defaultdict

# ─── 설정 ───
SEED = 42
RAW_PATH = os.path.join("data", "raw", "concat_names_mapped.csv")
OUTPUT_PATH = os.path.join("data", "raw", "concat_names_augmented.csv")
TRAIN_RATIO = 0.8
TARGET_COUNT = 400  # F1=0% 데이터부족 클래스 26개 전수 커버
ALLOWED_CHARS = list('abcdefghijklmnopqrstuvwxyz')

random.seed(SEED)


# ─── Augmentation 연산 ───

def replace_disallowed_chars(text):
    return ''.join(c.lower() if c.lower() in ALLOWED_CHARS + [' '] else '' for c in text)


def augment_name(name):
    """Character-level 변형 1회 적용."""
    if len(name) < 2:
        return name

    op = random.choice(['swap', 'delete', 'insert', 'substitute'])
    chars = list(name)

    if op == 'swap':
        candidates = [i for i in range(len(chars) - 1)
                       if chars[i] != ' ' and chars[i + 1] != ' ']
        if candidates:
            i = random.choice(candidates)
            chars[i], chars[i + 1] = chars[i + 1], chars[i]

    elif op == 'delete':
        candidates = [i for i, c in enumerate(chars) if c != ' ']
        if len(candidates) > 1:
            chars.pop(random.choice(candidates))

    elif op == 'insert':
        i = random.randint(0, len(chars))
        chars.insert(i, random.choice(ALLOWED_CHARS))

    elif op == 'substitute':
        candidates = [i for i, c in enumerate(chars) if c != ' ']
        if candidates:
            i = random.choice(candidates)
            replacement = random.choice([c for c in ALLOWED_CHARS if c != chars[i]])
            chars[i] = replacement

    return ''.join(chars)


# ─── 데이터 로딩 ───

def load_csv(path):
    """CSV를 읽어 (header, rows) 반환."""
    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = [row for row in reader]
    return fieldnames, rows


# ─── Stratified Split ───

def stratified_split(rows, label_key='NOC', train_ratio=0.8):
    """NOC 기준 stratified split. train/val/test 반환."""
    groups = defaultdict(list)
    for row in rows:
        groups[row[label_key]].append(row)

    train, val, test = [], [], []

    for label, group in groups.items():
        random.shuffle(group)
        n = len(group)

        if n >= 3:
            n_train = max(1, int(n * train_ratio))
            remainder = group[n_train:]
            n_val = max(1, len(remainder) // 2)

            train.extend(group[:n_train])
            val.extend(remainder[:n_val])
            test.extend(remainder[n_val:])
        elif n == 2:
            train.append(group[0])
            test.append(group[1])
        else:
            train.append(group[0])

    return train, val, test


# ─── Augmentation ───

def augment_minority(train_rows, target_count, name_key='Name', label_key='NOC'):
    """소수 클래스를 target_count까지 augmentation으로 채움."""
    groups = defaultdict(list)
    for row in train_rows:
        groups[row[label_key]].append(row)

    augmented = []
    stats = {}

    for label, group in groups.items():
        n = len(group)
        if n >= target_count:
            continue

        needed = target_count - n
        new_rows = []
        for _ in range(needed):
            src = random.choice(group)
            new_row = dict(src)
            new_row[name_key] = augment_name(new_row[name_key])
            new_rows.append(new_row)

        augmented.extend(new_rows)
        stats[label] = {'original': n, 'added': needed, 'ratio': f'{(n + needed) / n:.1f}x'}

    return augmented, stats


# ─── 메인 ───

def main():
    print(f"[1/4] 데이터 로딩: {RAW_PATH}")
    fieldnames, rows = load_csv(RAW_PATH)

    # 결측 제거
    rows = [r for r in rows if r.get('Name') and r.get('NOC') and r.get('LanguageGroup')]

    # 전처리
    for r in rows:
        r['Name'] = replace_disallowed_chars(r['Name'])
    rows = [r for r in rows if len(r['Name']) > 0]

    print(f"  전체 샘플: {len(rows)}")
    print(f"  클래스 수: {len(set(r['NOC'] for r in rows))}")

    print(f"\n[2/4] Stratified split (train {TRAIN_RATIO:.0%} / val 10% / test 10%)")
    train, val, test = stratified_split(rows, train_ratio=TRAIN_RATIO)
    print(f"  Train: {len(train)} | Val: {len(val)} | Test: {len(test)}")

    # Train 분포 확인
    train_counts = defaultdict(int)
    for r in train:
        train_counts[r['NOC']] += 1
    sorted_counts = sorted(train_counts.values())
    n_below = sum(1 for c in sorted_counts if c < TARGET_COUNT)
    print(f"  Train 최소: {min(sorted_counts)} | 최대: {max(sorted_counts)} | "
          f"target({TARGET_COUNT}) 미만: {n_below}개 클래스")

    print(f"\n[3/4] 소수 클래스 augmentation (target={TARGET_COUNT})")
    augmented_rows, stats = augment_minority(train, TARGET_COUNT)

    print(f"  대상 클래스: {len(stats)}개")
    print(f"  합성 샘플: +{len(augmented_rows)}건")
    print(f"  증강 후 Train: {len(train) + len(augmented_rows)}건")

    # 증강 통계 출력
    print(f"\n  {'클래스':<8} {'원본':>6} {'추가':>6} {'증강비':>8}")
    print(f"  {'─'*32}")
    for label in sorted(stats.keys()):
        s = stats[label]
        print(f"  {label:<8} {s['original']:>6} {s['added']:>6} {s['ratio']:>8}")

    # 합치기: augmented train + original val + original test
    # is_augmented 컬럼 추가로 구분 가능
    all_rows = []

    for r in train:
        r_copy = dict(r)
        r_copy['split'] = 'train'
        r_copy['is_augmented'] = 'false'
        all_rows.append(r_copy)

    for r in augmented_rows:
        r_copy = dict(r)
        r_copy['split'] = 'train'
        r_copy['is_augmented'] = 'true'
        all_rows.append(r_copy)

    for r in val:
        r_copy = dict(r)
        r_copy['split'] = 'val'
        r_copy['is_augmented'] = 'false'
        all_rows.append(r_copy)

    for r in test:
        r_copy = dict(r)
        r_copy['split'] = 'test'
        r_copy['is_augmented'] = 'false'
        all_rows.append(r_copy)

    # 저장
    out_fields = fieldnames + ['split', 'is_augmented']
    print(f"\n[4/4] 저장: {OUTPUT_PATH}")
    with open(OUTPUT_PATH, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=out_fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"  총 {len(all_rows)}건 저장 완료")
    print(f"  (train: {len(train) + len(augmented_rows)} | val: {len(val)} | test: {len(test)})")
    print(f"\n  컬럼 설명:")
    print(f"    split: train/val/test 구분")
    print(f"    is_augmented: true=합성 데이터, false=원본")


if __name__ == '__main__':
    main()
