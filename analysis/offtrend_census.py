# -*- coding: utf-8 -*-
"""off-trend 전수조사 (리버탈 대비 아티팩트).

두 개의 필터로 "논문의 13쌍 밖에 남는 off-trend 쌍"을 전부 나열한다.
  1) low-cnl 상위: cnl < 0.15 중 임베딩 유사도 상위 30
     → 검증 결과 30쌍 전원 col=1 (공식어 공유). 제도 층위가 잡는 쌍들.
  2) no-index: col=0 & cnl<0.15 & csl<0.3 & lpn<0.5 (어떤 지표에도 안 보임)
     → 상위가 필리핀x중남미(0.86-0.91), AFG-PAK(0.90), BAN-PAK(0.88).
       등록부 이식/종교 이름 스톡 메커니즘의 반복.

실행: 저장소 루트에서 python analysis/offtrend_census.py
출력: outputs/analysis/offtrend_census.csv
"""
import os
import sys

import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(BASE, 'analysis'))
from noc_iso3_mapping import noc_to_iso3

PAIRS_CSV = os.path.join(BASE, 'outputs', 'analysis', 'table1_rho_ci', 'pairs_ours.csv')
DICL_CSV = os.path.join(BASE, 'data', 'linguistics', 'DICL_v2.csv')
OUT_CSV = os.path.join(BASE, 'outputs', 'analysis', 'offtrend_census.csv')

LOW_CNL_CUT = 0.15   # 논문의 political-split 컷과 동일
TOP_N = 30
NOINDEX_CSL = 0.3
NOINDEX_LPN = 0.5
NOINDEX_SIM = 0.5


def build_paccording_table():
    pairs = pd.read_csv(PAIRS_CSV)
    dicl = pd.read_csv(DICL_CSV)

    dicl['key'] = ['|'.join(sorted([a, b]))
                   for a, b in zip(dicl['iso3_i'], dicl['iso3_j'])]
    dicl_u = dicl.drop_duplicates('key').set_index('key')

    pairs['iso_i'] = pairs['noc_i'].map(noc_to_iso3)
    pairs['iso_j'] = pairs['noc_j'].map(noc_to_iso3)
    pairs = pairs.dropna(subset=['iso_i', 'iso_j']).copy()
    pairs['key'] = ['|'.join(sorted([a, b]))
                    for a, b in zip(pairs['iso_i'], pairs['iso_j'])]
    for col in ['col', 'cor', 'cal', 'csl', 'lpn', 'lpa', 'bpn', 'bps']:
        pairs[col] = pairs['key'].map(dicl_u[col])
    return pairs


def main():
    pairs = build_paccording_table()

    low_cnl = pairs[pairs['cnl'] < LOW_CNL_CUT].sort_values(
        'emb_sim', ascending=False)
    top30 = low_cnl.head(TOP_N).copy()
    top30['filter'] = f'low_cnl_top{TOP_N}'

    noindex = pairs[(pairs['col'].fillna(0) == 0)
                    & (pairs['cnl'] < LOW_CNL_CUT)
                    & (pairs['csl'].fillna(0) < NOINDEX_CSL)
                    & (pairs['lpn'].fillna(0) < NOINDEX_LPN)
                    & (pairs['emb_sim'] > NOINDEX_SIM)].sort_values(
        'emb_sim', ascending=False).copy()
    noindex['filter'] = 'no_index_sim>0.5'

    out_cols = ['filter', 'noc_i', 'noc_j', 'emb_sim', 'cnl',
                'col', 'csl', 'lpn', 'bpn']
    out = pd.concat([top30, noindex])[out_cols]
    out.to_csv(OUT_CSV, index=False, encoding='utf-8')

    n_col1 = int((top30['col'] == 1).sum())
    print(f'low-cnl top{TOP_N}: {n_col1}/{TOP_N} share an official language')
    print(f'no-index pairs (sim>{NOINDEX_SIM}): {len(noindex)}')
    print('no-index top 12:')
    for _, r in noindex.head(12).iterrows():
        print(f"  {r['noc_i']}-{r['noc_j']} sim={r['emb_sim']:.3f}")
    print(f'saved: {OUT_CSV}')


if __name__ == '__main__':
    main()
