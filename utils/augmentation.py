"""
Character-level name augmentation for minority class oversampling.

전략:
  - swap:       인접한 두 문자 교환 (garcia → garica)
  - delete:     임의 문자 하나 삭제 (nguyen → nguen)
  - insert:     임의 위치에 알파벳 삽입 (kumar → kuamar)
  - substitute: 임의 문자 하나를 다른 알파벳으로 대체 (smith → snith)

음운론적 특성(n-gram 패턴)을 대부분 보존하면서도 약간의 변이를 주어
모델이 과적합하지 않게 돕는다.
"""

import random

import pandas as pd

ALLOWED_CHARS = list('abcdefghijklmnopqrstuvwxyz')


def augment_name(name: str) -> str:
    """이름에 character-level 변형 1회 적용."""
    if len(name) < 2:
        return name

    op = random.choice(['swap', 'delete', 'insert', 'substitute'])
    chars = list(name)

    if op == 'swap':
        # 공백이 아닌 인접 쌍 중 하나를 교환
        candidates = [
            i for i in range(len(chars) - 1)
            if chars[i] != ' ' and chars[i + 1] != ' '
        ]
        if candidates:
            i = random.choice(candidates)
            chars[i], chars[i + 1] = chars[i + 1], chars[i]

    elif op == 'delete':
        # 공백이 아닌 문자 하나 삭제 (이름 구조 유지)
        candidates = [i for i, c in enumerate(chars) if c != ' ']
        if len(candidates) > 1:  # 최소 1글자는 남겨야 함
            chars.pop(random.choice(candidates))

    elif op == 'insert':
        # 임의 위치에 알파벳 삽입
        i = random.randint(0, len(chars))
        chars.insert(i, random.choice(ALLOWED_CHARS))

    elif op == 'substitute':
        # 공백이 아닌 문자 하나를 다른 알파벳으로 대체
        candidates = [i for i, c in enumerate(chars) if c != ' ']
        if candidates:
            i = random.choice(candidates)
            # 원래 문자와 다른 문자로 대체
            replacement = random.choice([c for c in ALLOWED_CHARS if c != chars[i]])
            chars[i] = replacement

    return ''.join(chars)


def augment_minority_classes(
    df: pd.DataFrame,
    target_count: int,
    name_col: str = 'Name',
    label_col: str = 'NOC',
) -> pd.DataFrame:
    """
    소수 클래스를 target_count까지 augmentation으로 채움.
    target_count 이상인 클래스는 건드리지 않는다.

    Args:
        df: 학습 DataFrame (Name, NOC, LanguageGroup 포함)
        target_count: 이 수까지 augmentation으로 채움
        name_col: 이름 컬럼명
        label_col: 기준 라벨 컬럼명 (NOC)

    Returns:
        augmented DataFrame (원본 + 합성 행)
    """
    augmented_rows = []

    for label, group in df.groupby(label_col):
        n = len(group)
        if n >= target_count:
            continue

        needed = target_count - n
        for _ in range(needed):
            # 원본에서 하나 랜덤 샘플링 후 이름만 변형
            row = group.sample(1).iloc[0].copy()
            row[name_col] = augment_name(row[name_col])
            augmented_rows.append(row)

    if not augmented_rows:
        return df

    aug_df = pd.DataFrame(augmented_rows)
    result = pd.concat([df, aug_df], ignore_index=True)

    n_original = len(df)
    n_augmented = len(augmented_rows)
    n_classes_augmented = len(set(r[label_col] for _, r in pd.DataFrame(augmented_rows).iterrows()))
    print(
        f"[Augmentation] {n_classes_augmented}개 소수 클래스에 "
        f"{n_augmented}건 추가 (원본 {n_original} → 총 {len(result)})"
    )

    return result
