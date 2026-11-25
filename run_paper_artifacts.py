"""
논문 산출물 전체 파이프라인 — 한 번에 실행.

실행하는 5단계:
  1. DICL 상관분석 → Figure 2 데이터 + Appendix Table A1 생성
  2. Attention n-gram 추출 → Table 2 중간 CSV
  3. Figure 전체 생성 (t-SNE + DICL scatter + attention heatmap)
  4. Table 2 (attention 기반) 최종 CSV
  5. Table 2 비교용 frequency baseline CSV

Usage:
    # 기본값 (config의 inference.model_path 사용)
    python run_paper_artifacts.py

    # 모델 경로 명시
    python run_paper_artifacts.py --model_path outputs/checkpoints/best_model.pt

    # 일부 단계 건너뛰기 (이미 생성된 중간 산출물 재사용)
    python run_paper_artifacts.py --skip dicl_correlation extract_ngram

    # dry-run — 명령어만 출력
    python run_paper_artifacts.py --dry_run

단계별 의존성 (중요):
  step 1  → step 3의 Figure 2가 의존
  step 2  → step 4가 의존
  step 3  → 1번 없이 실행 시 Figure 2만 skip
  step 4  → 2번 없으면 fail
  step 5  → 독립 (raw CSV만 필요)
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import List


@dataclass
class Step:
    key: str
    title: str
    command: List[str]
    depends_on_model: bool = True   # 체크포인트 필요 여부


def build_steps(config: str, model_path: str | None,
                samples_per_group: int, min_coverage: float) -> List[Step]:
    """5단계 파이프라인 정의."""
    model_args = ['--model_path', model_path] if model_path else []

    return [
        Step(
            key='dicl_correlation',
            title='1/5 DICL 상관분석 (Figure 2 data + Table A1)',
            command=[
                sys.executable, '-m', 'analysis.dicl_correlation',
                '--config', config,
                *model_args,
            ],
            depends_on_model=True,
        ),
        Step(
            key='extract_ngram',
            title='2/5 Attention n-gram CSV 추출 (Table 2 중간 데이터)',
            command=[
                sys.executable, '-m', 'analysis.visualize', 'extract_ngram',
                '--config', config,
                *model_args,
                '--samples_per_group', str(samples_per_group),
            ],
            depends_on_model=True,
        ),
        Step(
            key='visualize_all',
            title='3/5 Figure 전체 생성 (t-SNE + DICL scatter + attention heatmap)',
            command=[
                sys.executable, '-m', 'analysis.visualize', 'all',
                '--config', config,
                *(['--model_path', model_path] if model_path else []),
            ],
            depends_on_model=True,
        ),
        Step(
            key='table_attention',
            title='4/5 Table 2 (attention 기반) CSV 생성',
            command=[
                sys.executable, '-m', 'analysis.build_top_ngram_table',
                '--mode', 'attention',
                '--min_coverage', str(min_coverage),
            ],
            depends_on_model=False,
        ),
        Step(
            key='table_frequency',
            title='5/5 Frequency baseline CSV 생성 (본문 2문장 근거)',
            command=[
                sys.executable, '-m', 'analysis.build_top_ngram_table',
                '--mode', 'frequency',
                '--min_coverage', str(min_coverage),
            ],
            depends_on_model=False,
        ),
    ]


def run_step(step: Step, dry_run: bool) -> tuple[bool, float]:
    """단일 단계 실행. (성공여부, 소요시간) 반환."""
    cmd_str = ' '.join(step.command)
    print("\n" + "=" * 72)
    print(f"  [{step.title}]")
    print(f"  $ {cmd_str}")
    print("=" * 72)

    if dry_run:
        print("  [dry-run] 실행 생략")
        return True, 0.0

    start = time.time()
    try:
        # stream 출력 유지 — 긴 단계의 진행상황 실시간 확인
        result = subprocess.run(step.command, check=False)
        elapsed = time.time() - start
        success = (result.returncode == 0)
        if success:
            print(f"\n  ✓ 완료 ({elapsed:.1f}s)")
        else:
            print(f"\n  ✗ 실패: return code {result.returncode} ({elapsed:.1f}s)")
        return success, elapsed
    except KeyboardInterrupt:
        elapsed = time.time() - start
        print(f"\n  ⚠ 사용자 중단 ({elapsed:.1f}s)")
        raise


def main():
    parser = argparse.ArgumentParser(
        description='논문 산출물 전체 파이프라인 실행',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument('--config', default='configs/base_config.yaml',
                        help='config YAML 경로')
    parser.add_argument('--model_path', default=None,
                        help='체크포인트 경로 (미지정 시 config.inference.model_path 사용)')
    parser.add_argument('--samples_per_group', type=int, default=500,
                        help='attention 추출 시 언어권당 샘플 수 (기본 500)')
    parser.add_argument('--min_coverage', type=float, default=0.03,
                        help='Table 2에 적용할 coverage 하한 (기본 0.03)')
    parser.add_argument('--skip', nargs='+', default=[],
                        choices=['dicl_correlation', 'extract_ngram',
                                 'visualize_all', 'table_attention',
                                 'table_frequency'],
                        help='건너뛸 단계 key (공백 구분 나열 가능)')
    parser.add_argument('--only', nargs='+', default=None,
                        choices=['dicl_correlation', 'extract_ngram',
                                 'visualize_all', 'table_attention',
                                 'table_frequency'],
                        help='지정된 단계만 실행 (--skip과 상호 배타)')
    parser.add_argument('--dry_run', action='store_true',
                        help='명령어만 출력하고 실제 실행은 안 함')
    parser.add_argument('--stop_on_error', action='store_true',
                        help='첫 실패 시 즉시 중단 (기본: 다음 단계 진행)')
    args = parser.parse_args()

    # 단계 구성 + 필터링
    all_steps = build_steps(
        config=args.config, model_path=args.model_path,
        samples_per_group=args.samples_per_group,
        min_coverage=args.min_coverage,
    )

    if args.only:
        steps = [s for s in all_steps if s.key in args.only]
    else:
        steps = [s for s in all_steps if s.key not in args.skip]

    if not steps:
        print("실행할 단계가 없음.")
        return 0

    # 헤더
    print("\n" + "#" * 72)
    print("# 논문 산출물 파이프라인")
    print(f"# config       : {args.config}")
    print(f"# model_path   : {args.model_path or '(config에서 자동)'}")
    print(f"# samples/group: {args.samples_per_group}")
    print(f"# min_coverage : {args.min_coverage}")
    print(f"# 실행 단계    : {[s.key for s in steps]}")
    if args.dry_run:
        print("# DRY-RUN 모드 (실제 실행 안 함)")
    print("#" * 72)

    # 순차 실행
    results = []
    total_start = time.time()
    for step in steps:
        try:
            success, elapsed = run_step(step, args.dry_run)
        except KeyboardInterrupt:
            print("\n사용자 중단 — 지금까지 완료된 단계 요약:")
            break
        results.append((step, success, elapsed))
        if not success and args.stop_on_error:
            print("\n--stop_on_error 설정 — 파이프라인 중단.")
            break
    total_elapsed = time.time() - total_start

    # 최종 요약
    print("\n" + "#" * 72)
    print("# 실행 결과 요약")
    print("#" * 72)
    for step, success, elapsed in results:
        mark = "✓" if success else "✗"
        print(f"  {mark} {step.key:20s}  {elapsed:6.1f}s")
    print(f"\n  총 소요: {total_elapsed:.1f}s")

    # 주요 산출물 경로 안내
    print("\n[생성된 산출물 위치]")
    print("  outputs/figures/")
    print("    ├─ tsne_embeddings.pdf          (Figure 1)")
    print("    ├─ dicl_scatter.pdf             (Figure 2)")
    print("    ├─ attention_heatmap.pdf        (Appendix, 선택)")
    print("    └─ top_{bi,tri}grams_by_lang_group.csv  (중간)")
    print("  outputs/tables/")
    print("    ├─ dicl_metric_comparison.csv           (Appendix Table A1)")
    print("    ├─ top_ngram_table_attention_cov0.03.csv (Table 2)")
    print("    └─ top_ngram_table_frequency_cov0.03.csv (baseline)")

    # 실패 있으면 non-zero exit code
    any_fail = any(not s for _, s, _ in results)
    return 1 if any_fail else 0


if __name__ == '__main__':
    sys.exit(main())
