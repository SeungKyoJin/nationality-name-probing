"""
analysis/nameprism_membership.py

NamePrism 118개국 리스트 (Ye et al. 2017, Figure 5 treemap에서 추출) 기반으로
데이터셋 168개 NOC를 두 그룹으로 분할:
  · nameprism  : NamePrism CEL taxonomy에 직접 등재 → 라벨을 그대로 상속 (113개국)
  · lpn        : 미등재 → DICL LPN 최근접 국가의 라벨 상속 (55개국)

rebuttal R2 (Reviewer HUuH 순환성 대응): "독립적으로 라벨된 113개국만으로
결과를 보고하라" 요청의 선행 작업. paper Appendix "Language-group label
assignment" 문단의 2단계 절차와 동일 기준.

출처: data/papers/'Nationality Classification Using Name Embeddings.pdf'
      p.6 Figure 5 (pypdf 텍스트 추출, 2026-09-17 재확인)

실행:
  python -m analysis.nameprism_membership
"""

from __future__ import annotations

import argparse
import os

import pandas as pd
import yaml

from analysis.noc_iso3_mapping import noc_to_iso3

# Ye et al. (2017) Figure 5의 39개 leaf × 소속 국가 → ISO3.
# 표기 그대로 옮기되 ISO3로 정규화 (Hungry→HUN, Portual→PRT, Netherland→NLD 등
# 원문 오탈자 포함 확인).
NAMEPRISM_LEAVES = {
    'SouthAsian': ['NPL', 'IND', 'LKA'],
    'CelticEnglish': ['IRL', 'GBR'],
    'Jewish': ['ISR'],
    'Greek': ['GRC'],
    'ArabianPeninsula': ['ARE', 'IRQ', 'YEM', 'BHR', 'SYR', 'JOR', 'OMN',
                         'LBN', 'QAT', 'KWT', 'SAU'],
    'Maghreb': ['TUN', 'MAR', 'DZA', 'LBY'],
    'Nubian': ['EGY', 'SOM', 'SDN'],
    'Persian': ['IRN', 'AFG'],
    'Pakistan': ['PAK'],
    'Bangladesh': ['BGD'],
    'Turkey': ['TUR'],
    'CentralAsian': ['KAZ', 'AZE', 'UZB'],
    'EastEuropean': ['CZE', 'SVK', 'HUN', 'POL'],
    'SouthSlavs': ['BIH', 'SRB', 'SVN', 'MNE', 'BGR', 'HRV', 'MKD'],
    'German': ['DEU', 'AUT', 'NLD', 'CHE'],
    'Baltics': ['LVA', 'EST', 'LTU'],
    'Russian': ['RUS', 'BLR', 'UKR'],
    'French': ['BEL', 'FRA'],
    'Italy': ['ITA'],
    'Romania': ['ROU'],
    'WestAfrican': ['COG', 'LBR', 'SEN', 'SLE', 'GIN', 'GHA', 'TGO', 'BEN',
                    'NGA'],
    'EastAfrican': ['ZMB', 'RWA', 'TZA', 'KEN', 'SSD', 'ETH', 'UGA'],
    'SouthAfrican': ['BWA', 'NAM', 'ZWE', 'ZAF'],
    'Spanish': ['DOM', 'URY', 'GTM', 'COL', 'VEN', 'CHL', 'PAN', 'BOL',
                'SLV', 'ECU', 'ARG', 'HND', 'PER', 'CRI', 'PRY', 'MEX',
                'ESP', 'NIC'],
    'Portuguese': ['PRT', 'BRA', 'AGO', 'MOZ'],
    'Philippines': ['PHL'],
    'Chinese': ['SGP', 'HKG', 'CHN', 'TWN'],
    'SouthKorea': ['KOR'],
    'Japan': ['JPN'],
    'Cambodia': ['KHM'],
    'Thailand': ['THA'],
    'Myanmar': ['MMR'],
    'Vietnam': ['VNM'],
    'Indonesia': ['IDN'],
    'Malaysia': ['MYS'],
    'Denmark': ['DNK'],
    'Sweden': ['SWE'],
    'Norway': ['NOR'],
    'Finland': ['FIN'],
}

NAMEPRISM_ISO3 = {iso for isos in NAMEPRISM_LEAVES.values() for iso in isos}


def build_membership(config) -> pd.DataFrame:
    """데이터셋 NOC 전체 → NamePrism 등재 여부 테이블."""
    df = pd.read_csv(config['data']['raw_path'], usecols=['NOC', 'LanguageGroup'])
    nocs = df.drop_duplicates('NOC')[['NOC', 'LanguageGroup']].copy()
    nocs['iso3'] = nocs['NOC'].apply(noc_to_iso3)
    nocs['label_source'] = nocs['iso3'].map(
        lambda i: 'nameprism' if i in NAMEPRISM_ISO3 else 'lpn')
    return nocs.sort_values('NOC').reset_index(drop=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', default=os.path.join('configs', 'base_config.yaml'))
    p.add_argument('--output',
                   default=os.path.join('outputs', 'analysis',
                                        'nameprism_membership.csv'))
    args = p.parse_args()

    with open(args.config, encoding='utf-8') as f:
        config = yaml.safe_load(f)

    table = build_membership(config)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    table.to_csv(args.output, index=False)

    n_np = int((table['label_source'] == 'nameprism').sum())
    n_lpn = int((table['label_source'] == 'lpn').sum())
    print(f'전체 NOC: {len(table)}  |  NamePrism 등재: {n_np}  |  LPN 배정: {n_lpn}')
    print(f'(paper 기준 기대값: 113 / 55)')
    print(f'\nLPN 배정 NOC ({n_lpn}개):')
    print(', '.join(table.loc[table["label_source"] == "lpn", "NOC"]))
    print(f'\n저장: {args.output}')


if __name__ == '__main__':
    main()
