"""
언어권별 Top-k tri-gram 테이블 (Table 1) 생성기.

두 모드 지원 — 동일 레이아웃, 입력 데이터만 다름:
  - frequency: 원본 CSV에서 단순 빈도 기준 top-k (baseline, 체크포인트 불필요)
  - attention: visualize.py의 extract_ngram 결과 CSV에서 평균 attention 기준 top-k (논문용)

선정 기준 (사전 정의, cherry-picking 방지):
  1. 테스트셋 language-group macro-F1 ≥ 0.80
  2. 언어 계통(family) diversity — 계통 중복 시 F1 상위 하나만 유지
  → 7개 언어권 확정 (Myanmar는 Chinese와 Sino-Tibetan 중복으로 제외)

Usage:
    # 지금 바로 돌릴 수 있음 (baseline 미리보기)
    python -m analysis.build_top_ngram_table --mode frequency

    # 모델 체크포인트로 attention CSV 생성 후 (논문용)
    python -m analysis.visualize extract_ngram
    python -m analysis.build_top_ngram_table --mode attention

    # 샘플 500개 기준 coverage 3% 이상 필터 적용
    python -m analysis.build_top_ngram_table --mode attention --min_coverage 0.03

출력:
  outputs/tables/top_ngram_table_{mode}[_cov0.XX].csv
    컬럼: rank, language_group, family, f1, ngram, coverage,
          linguistic_interpretation, (count | mean_attn)
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import Counter
from typing import Iterable

import pandas as pd


# ─── 선정된 7개 언어권 + 계통 ───

SELECTED_GROUPS = [
    # (lang_group, language_family)
    ('SouthKorea',    'Koreanic'),
    ('Japan',         'Japonic'),
    ('Vietnam',       'Austroasiatic'),
    ('Finland',       'Uralic'),
    ('Thailand',      'Kra-Dai'),
    ('Chinese',       'Sino-Tibetan'),
    ('EastEuropean',  'Indo-European (Slavic)'),
]

# 기본 경로 — 프로젝트 루트 기준
DEFAULT_RAW_CSV = 'data/raw/concat_names_mapped.csv'
DEFAULT_LANG_F1_CSV = 'outputs/260421/class_report_lang_model_multi_a0.5_1.pt.csv'
DEFAULT_ATTN_CSV = 'outputs/figures/top_trigrams_by_lang_group.csv'
DEFAULT_OUTPUT_DIR = 'outputs/tables'

# 허용 문자 — utils/util.py의 replace_disallowed_chars와 동일 규칙
_ALLOWED_CHARS = set('abcdefghijklmnopqrstuvwxyz ')


# ─── 언어학적 해석 (사전 정의) ───
# 논문 부록/본문 Table 3에 들어갈 linguistic_interpretation 컬럼을 자동 채움.
# 모델이 attention을 둔 trigram이 어떤 언어학적 패턴인지 짧게 명시.
# 주의: 정성적 해석이므로 데이터에서 자동 derive 불가 — 수기 사전 매핑.

LINGUISTIC_INTERPRETATIONS = {
    # ─ Koreanic (South Korea) ─
    'eon': 'Korean syllabic block (e.g., 연/은/원)',
    'hoi': 'Korean syllable (회) — common in given names',
    'son': '손 (Sohn surname) / closed syllable -onsuffix',
    'lee': '이 (Lee/Yi) — most common Korean surname',
    'eun': '은 (eun) — frequent given-name syllable',
    # ─ Japonic (Japan) ─
    'oji': 'Japanese name ending (e.g., Senji, Koji)',
    'azu': 'Japanese morpheme (e.g., Azuma, Kazu-)',
    'aki': '-aki ending (Aki, Maki, Akihito)',
    'awa': '-awa toponymic suffix (Tagawa, Asakawa)',
    'ano': '-ano surname ending (Sano, Hirano, Asano)',
    # ─ Austroasiatic (Vietnam) ─
    'guy': 'Nguy(ễn) — Vietnam\'s most common surname prefix',
    'tri': 'Vietnamese given-name element (Trí)',
    'tra': 'Tr(ần) — second most common Vietnamese surname',
    'hoa': 'Hoa (花) — common name element',
    'goc': 'Vietnamese diacritic-stripped form',
    # ─ Uralic (Finland) ─
    'ain': 'Finnish suffix component (-ainen lead)',
    'avi': 'Finnish given-name element (David, Aapo)',
    'tti': 'Finnish gemination -tti (Niitti, Kotti)',
    'art': 'Finnish loan/personal name (Arttu, Marttinen)',
    'ola': '-ola/-ölä surname suffix (Niskola, Pajala)',
    # ─ Kra-Dai (Thailand) ─
    'awa': 'Pali/Sanskrit-derived element',  # NOTE: shared with Japan
    'asa': 'Pali-derived (Asaree, Asawin)',
    'ano': 'Royal/honorific suffix in Thai names',  # shared with Japan
    'att': 'Thai given-name component',
    'anu': '-anu (Anu-, Anuwat) — common Thai prefix',
    # ─ Sino-Tibetan (Chinese) ─
    'ian': 'Mandarin pinyin syllable (-ian: tian, lian)',
    'zha': 'Pinyin onset zh- + a (Zhang, Zhao)',
    'xia': 'Pinyin syllable xia (Xia surname, -xia)',
    'ing': '-ing pinyin ending (Ming, Ying, Bing)',
    'ang': '-ang pinyin ending (Wang, Zhang, Yang)',
    # ─ Indo-European Slavic (East European) ─
    'rti': 'Slavic consonant cluster (Bartoš, Martinov)',
    'art': 'Slavic name root (Bart-, Mart-)',  # shared with Finnish
    'ova': 'Slavic feminine patronymic ending (-ová, -ova)',
    'jan': 'Slavic given name (Jan/Ján)',
    'are': 'Slavic surname segment',
}


def get_interpretation(ngram: str, language_group: str = '') -> str:
    """
    Trigram에 대한 언어학적 해석 반환. 매핑 없으면 빈 문자열.

    Note: 일부 trigram(awa, ano, art)이 여러 언어권에 등장 — language_group으로
    구분 가능하지만 현재 단순 lookup만 수행. 필요 시 (ngram, lang_group) 튜플
    매핑으로 확장.
    """
    return LINGUISTIC_INTERPRETATIONS.get(ngram, '')


# ─── 공통 유틸 ───

def clean_name(text: str) -> str:
    """소문자 변환 + 허용 문자만 유지 — 모델 학습 전처리와 동일."""
    if not isinstance(text, str):
        return ''
    return ''.join(c for c in text.lower() if c in _ALLOWED_CHARS)


def load_language_f1(csv_path: str) -> dict:
    """
    class_report_lang_*.csv 에서 언어권별 F1 점수 로드.
    accuracy/macro avg/weighted avg 행 제외.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"F1 CSV 없음: {csv_path}")
    df = pd.read_csv(csv_path, index_col=0)
    # 첫 컬럼이 인덱스 → 언어권명. 'accuracy', 'macro avg', 'weighted avg' 제외
    skip = {'accuracy', 'macro avg', 'weighted avg'}
    return {name: row['f1-score'] for name, row in df.iterrows() if name not in skip}


# ─── 모드 A: frequency (원본 CSV에서 빈도 기준) ───

def extract_top_ngrams_by_frequency(raw_csv_path: str, ngram_n: int = 3,
                                     top_k: int = 5, min_coverage: float = 0.0) -> dict:
    """
    원본 CSV에서 언어권별 tri-gram 빈도를 집계하고 top-k 반환.

    Attention 없이 단순 빈도 기반 — 모델이 없어도 실행 가능한 baseline.
    공백 포함 n-gram은 성/이름 경계에 걸친 것으로 간주하여 제외.
    min_coverage 미만의 n-gram은 필터링 (통계적 안정성 확보).

    Returns: {lang_group: [(ngram, count, coverage), ...]}
        coverage = (해당 n-gram 포함 이름 수) / (언어권 샘플 수)
    """
    if not os.path.exists(raw_csv_path):
        raise FileNotFoundError(f"원본 CSV 없음: {raw_csv_path}")

    df = pd.read_csv(raw_csv_path)
    required = {'Name', 'LanguageGroup'}
    if not required.issubset(df.columns):
        raise ValueError(f"필수 컬럼 누락. 실제: {list(df.columns)}")

    df = df.dropna(subset=['Name', 'LanguageGroup']).copy()
    df['Name_clean'] = df['Name'].apply(clean_name)
    df = df[df['Name_clean'].str.len() >= ngram_n]

    result = {}
    for lang, group in df.groupby('LanguageGroup'):
        names = group['Name_clean'].tolist()
        ngram_count = Counter()
        ngram_name_set = {}  # ngram → set(name_idx) for coverage 계산

        for idx, name in enumerate(names):
            seen_this_name = set()
            for j in range(len(name) - ngram_n + 1):
                ng = name[j:j + ngram_n]
                if ' ' in ng:
                    continue
                ngram_count[ng] += 1
                if ng not in seen_this_name:
                    ngram_name_set.setdefault(ng, 0)
                    ngram_name_set[ng] += 1
                    seen_this_name.add(ng)

        n_samples = len(names)
        # 빈도순 전체 랭킹 → coverage 필터 → 상위 top_k
        ranked = []
        for ng, cnt in ngram_count.most_common():
            cov = ngram_name_set[ng] / n_samples if n_samples else 0.0
            if cov < min_coverage:
                continue
            ranked.append((ng, cnt, cov))
            if len(ranked) >= top_k:
                break
        result[lang] = ranked
    return result


# ─── 모드 B: attention (CSV에서 평균 attention 기준) ───

def extract_top_ngrams_by_attention(attn_csv_path: str, top_k: int = 5,
                                     min_coverage: float = 0.0) -> dict:
    """
    visualize.py의 extract_ngram 결과 CSV에서 top-k 추출.
    CSV 컬럼: lang_group, ngram, avg_attn, coverage, freq

    평균 attention weight 기준 내림차순 top-k.
    min_coverage 미만의 n-gram은 제외 — 샘플 수 적을 때 rare outlier 배제.
      - 500 샘플: 0.03 권장 (15 샘플 이상 등장)
      - 150 샘플: 0.0 (기존 min_freq=5 필터 충분)

    Returns: {lang_group: [(ngram, avg_attn, coverage), ...]}
    """
    if not os.path.exists(attn_csv_path):
        raise FileNotFoundError(
            f"Attention CSV 없음: {attn_csv_path}\n"
            f"먼저 `python -m analysis.visualize extract_ngram`을 실행하세요."
        )
    df = pd.read_csv(attn_csv_path)
    if min_coverage > 0:
        before = len(df)
        df = df[df['coverage'] >= min_coverage]
        print(f"[min_coverage={min_coverage:.2f}] {before} → {len(df)} n-grams")
    result = {}
    for lang, group in df.groupby('lang_group'):
        top = group.nlargest(top_k, 'avg_attn')
        result[lang] = [
            (r['ngram'], r['avg_attn'], r['coverage'])
            for _, r in top.iterrows()
        ]
    return result


# ─── 메인 파이프라인 ───

def _rows_to_dataframe(per_lang: dict, f1_map: dict, mode: str) -> pd.DataFrame:
    """
    언어권별 top-k 리스트를 long-format DataFrame으로 변환.
    한 행 = 하나의 (language_group, rank, n-gram) 조합.

    공통 컬럼: rank, language_group, family, f1, ngram, coverage, linguistic_interpretation
    모드별 컬럼:
      - frequency: count
      - attention: mean_attn
    """
    records = []
    for lang, family in SELECTED_GROUPS:
        if lang not in per_lang:
            continue
        for rank, item in enumerate(per_lang[lang], start=1):
            ng, value, cov = item
            record = {
                'rank': rank,
                'language_group': lang,
                'family': family,
                'f1': f1_map.get(lang),
                'ngram': ng,
                'coverage': round(cov, 4),
                'linguistic_interpretation': get_interpretation(ng, lang),
            }
            if mode == 'frequency':
                record['count'] = int(value)
            elif mode == 'attention':
                record['mean_attn'] = round(value, 4)
            records.append(record)
    return pd.DataFrame(records)


def build_table(mode: str, raw_csv: str, attn_csv: str, lang_f1_csv: str,
                top_k: int, min_coverage: float, output_dir: str) -> None:
    # 1. F1 로드
    try:
        f1_map = load_language_f1(lang_f1_csv)
    except FileNotFoundError as e:
        print(f"[WARN] F1 CSV 로드 실패 — F1 컬럼은 빈값\n  {e}", file=sys.stderr)
        f1_map = {}

    # 2. 모드별 데이터 추출
    if mode == 'frequency':
        per_lang = extract_top_ngrams_by_frequency(
            raw_csv, ngram_n=3, top_k=top_k, min_coverage=min_coverage,
        )
    elif mode == 'attention':
        per_lang = extract_top_ngrams_by_attention(
            attn_csv, top_k=top_k, min_coverage=min_coverage,
        )
    else:
        raise ValueError(f"mode는 frequency 또는 attention: {mode}")

    # 3. 누락 언어권 경고
    missing = [lang for lang, _ in SELECTED_GROUPS if lang not in per_lang]
    if missing:
        print(f"[WARN] 데이터에 없는 언어권: {missing}", file=sys.stderr)

    # 4. DataFrame 구성 + CSV 저장
    df = _rows_to_dataframe(per_lang, f1_map, mode)

    os.makedirs(output_dir, exist_ok=True)
    cov_tag = f'_cov{min_coverage:.2f}' if min_coverage > 0 else ''
    csv_path = os.path.join(output_dir, f'top_ngram_table_{mode}{cov_tag}.csv')
    df.to_csv(csv_path, index=False)

    print(f"\n✓ CSV 저장: {csv_path}  ({len(df)}행, 언어권 {df['language_group'].nunique()}개)")
    print(f"\n--- 미리보기 ({mode} mode, min_coverage={min_coverage}) ---")
    # 언어권별 top-5를 읽기 쉽게 표시
    for lang in df['language_group'].unique():
        sub = df[df['language_group'] == lang]
        score_col = 'mean_attn' if mode == 'attention' else 'count'
        items = [f"{r['ngram']}({r[score_col]}, cov={r['coverage']})"
                 for _, r in sub.iterrows()]
        print(f"  {lang:15s}: {', '.join(items)}")


# ─── CLI ───

def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument('--mode', choices=['frequency', 'attention'], required=True)
    parser.add_argument('--raw_csv', default=DEFAULT_RAW_CSV,
                        help='frequency 모드용 원본 CSV')
    parser.add_argument('--attn_csv', default=DEFAULT_ATTN_CSV,
                        help='attention 모드용 top_trigrams_by_lang_group.csv')
    parser.add_argument('--lang_f1_csv', default=DEFAULT_LANG_F1_CSV,
                        help='언어권 F1 점수 CSV (class_report_lang_*)')
    parser.add_argument('--top_k', type=int, default=5)
    parser.add_argument('--min_coverage', type=float, default=0.0,
                        help='최소 coverage (0.0=필터 없음). 500 샘플 기준 0.03 권장, '
                             '150 샘플 기준 0.0~0.02')
    parser.add_argument('--output_dir', default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    build_table(
        mode=args.mode,
        raw_csv=args.raw_csv,
        attn_csv=args.attn_csv,
        lang_f1_csv=args.lang_f1_csv,
        top_k=args.top_k,
        min_coverage=args.min_coverage,
        output_dir=args.output_dir,
    )


if __name__ == '__main__':
    main()
