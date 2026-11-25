"""
Top n-gram 테이블 재생성 — one-shot pipeline (cross-platform).

Usage:
    python -m analysis.regen_ngram
"""

import glob
import os
import subprocess
import sys

import yaml


def main():
    # 1. n_layers=3 임시 config 생성
    cfg = yaml.safe_load(open('configs/base_config.yaml', 'r', encoding='utf-8'))
    cfg['model']['n_layers'] = 3
    with open('configs/_anal_l3.yaml', 'w', encoding='utf-8') as f:
        yaml.dump(cfg, f)
    print('[1/4] config 생성: configs/_anal_l3.yaml (n_layers=3)')

    # 2. 체크포인트 + F1 csv 자동 탐색
    model_paths = glob.glob('outputs/checkpoints/**/conf_alpha_0.3.pt', recursive=True)
    f1_paths = glob.glob(
        'outputs/checkpoints/**/class_report_lang_conf_alpha_0.3.pt.csv',
        recursive=True,
    )
    if not model_paths:
        print('[ERROR] conf_alpha_0.3.pt 못 찾음. outputs/checkpoints/ 확인')
        sys.exit(1)
    if not f1_paths:
        print('[ERROR] class_report_lang_conf_alpha_0.3.pt.csv 못 찾음')
        sys.exit(1)

    model = model_paths[0]
    f1csv = f1_paths[0]
    print(f'[2/4] Model:  {model}')
    print(f'      F1CSV:  {f1csv}')

    # 3. attention CSV 추출
    print('[3/4] attention CSV 추출 중 (~30초)...')
    subprocess.run([
        sys.executable, '-m', 'analysis.visualize', 'extract_ngram',
        '--config', 'configs/_anal_l3.yaml',
        '--model_path', model,
        '--samples_per_group', '150',
    ], check=True)

    # 4. n-gram 테이블 생성
    print('[4/4] n-gram 테이블 생성 중...')
    subprocess.run([
        sys.executable, '-m', 'analysis.build_top_ngram_table',
        '--mode', 'attention',
        '--min_coverage', '0.03',
        '--lang_f1_csv', f1csv,
        '--attn_csv', 'outputs/figures/top_trigrams_by_lang_group.csv',
    ], check=True)

    # 5. 결과 출력
    out = 'outputs/tables/top_ngram_table_attention_cov0.03.csv'
    print()
    print(f'=== {out} ===')
    if os.path.exists(out):
        print(open(out, 'r', encoding='utf-8').read())
    else:
        print(f'[WARN] 결과 파일 없음: {out}')


if __name__ == '__main__':
    main()
