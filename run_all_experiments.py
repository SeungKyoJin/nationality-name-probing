"""
전체 실험 오케스트레이션 스크립트.

각 Step은 이전 Step의 최적 결과를 자동으로 참조한다.
한 번 실행하면 Step 0~6까지 자동으로 이어진다.

Usage:
    # 전체 파이프라인 (Step 0→6 자동 연결)
    python run_all_experiments.py --config configs/base_config.yaml

    # 특정 단계만 실행 (이전 결과가 experiment_results.csv에 있어야 함)
    python run_all_experiments.py --step grid_search
    python run_all_experiments.py --step ablation_alpha
    python run_all_experiments.py --step baselines

실행 흐름:
    Step 0: 그리드서치 (lr × n_layers × embedding_dim)
            → 최적 base config 확정
    Step 1: α ablation
            → 최적 α 확정
    Step 2: Channel ablation (uni/bi/tri/multi)
            → Table 2(b)
    Step 3: Aug+CW ablation
            → Table 2(c)
    Step 4: Max length ablation
            → Table 2(d)
    Step 5: Baseline 모델 비교
            → Table 1
    Step 6: DICL 분석 + 시각화
            → Figure 1, 2, 3
"""

import os
import copy
import time
import argparse
import itertools

import yaml
import pandas as pd

from utils.util import set_seed
from train import run_training


# ─── 유틸리티 ───

def run_step(config, step_name, overrides, save_prefix):
    """config를 override하여 단일 실험 실행. metrics 반환."""
    cfg = copy.deepcopy(config)
    for keys, value in overrides:
        section = cfg
        for k in keys[:-1]:
            section = section[k]
        section[keys[-1]] = value

    cfg['train']['save_name'] = f"{save_prefix}.pt"
    print(f"\n{'─' * 60}")
    print(f"  {step_name}: {save_prefix}  (seed={cfg['experiment']['seed']})")
    print(f"{'─' * 60}")

    # Seed re-application — overrides에서 experiment.seed 변경 시 반영
    # (multi-seed 실험: 같은 config로 다른 seed 실행 시 필수)
    set_seed(cfg['experiment']['seed'])

    t = time.time()
    try:
        metrics = run_training(cfg)
        elapsed = time.time() - t
        print(f"  완료 ({elapsed:.1f}s)")
        # 실험 조건도 함께 반환
        metrics['_save_prefix'] = save_prefix
        metrics['_elapsed'] = elapsed
        return metrics
    except Exception as e:
        print(f"  실패: {e}")
        return None


def select_best(results, metric_key, higher_is_better=True):
    """실험 결과 리스트에서 최적 결과를 선택."""
    valid = [(r, r[metric_key]) for r in results if r is not None]
    if not valid:
        raise RuntimeError("유효한 실험 결과가 없습니다.")
    if higher_is_better:
        best = max(valid, key=lambda x: x[1])
    else:
        best = min(valid, key=lambda x: x[1])
    return best[0]


def print_best(label, result, keys):
    """최적 결과 출력."""
    print(f"\n  ★ {label}: {result['_save_prefix']}")
    for k in keys:
        print(f"    {k}: {result[k]:.4f}" if isinstance(result[k], float) else f"    {k}: {result[k]}")


# ─── Step 5b: LR baseline ───

def step_lr_baseline(config):
    """LR baseline (sklearn, 별도 프로세스)."""
    print("\n" + "=" * 60)
    print(" Step 5b: LR Baseline")
    print("=" * 60)
    import subprocess
    subprocess.run([
        'python', 'run_baseline_lr.py',
        '--config', 'configs/base_config.yaml'
    ])


# ─── Clean Rerun: aug=False, cw=False 통일 16개 실험 ───

def step_clean_rerun(config):
    """
    Clean rerun under unified conditions (no aug, no class_weight, no synthetic).
    Total 16 experiments with sequential dependency for contrastive γ sweep.

    Tier 1 (9 exp): Baselines + Pool + Fusion + Combined
    Tier 2a (3 exp): Contrastive variants nat/lang/hier @ γ=0.3
    Tier 2b (2 exp): γ ∈ {0.1, 0.5} with best variant from Tier 2a
    Tier 3 (2 exp):  Synthetic data ablation (target=400, target=1000)
    """
    print("\n" + "=" * 60)
    print(" Clean Rerun (16 experiments)")
    print("   Conditions: aug=False, class_weight=False, synthetic=null")
    print("=" * 60)

    # 검증된 best HP (이전 noaug_nocw 47.43% 결과 기반)
    common = [
        (['data', 'augment'], False),
        (['train', 'class_weight'], False),
        (['data', 'synthetic_path'], None),
        (['train', 'alpha'], 0.5),
        (['train', 'learning_rate'], 5e-5),
        (['model', 'n_layers'], 4),
        (['model', 'embedding_dim'], 256),
        (['model', 'hidden_dim'], 1024),
        (['model', 'n_head'], 8),
        (['data', 'max_length'], 90),
        (['data', 'ngram'], 'multi'),
        (['train', 'contrastive_weight'], 0.0),  # default off
    ]

    # ─── Tier 1.1: Baselines (4) ───
    print("\n" + "─" * 60)
    print(" Tier 1.1: Baselines (4 experiments)")
    print("─" * 60)
    for model_type, label in [
        ('multichannel_cnn', 'clean_baseline_cnn'),
        ('multichannel_bilstm', 'clean_baseline_bilstm'),
        ('multichannel_fasttext', 'clean_baseline_fasttext'),
        ('multichannel_encoder', 'clean_ours_transformer'),
    ]:
        run_step(config, f"baseline={model_type}", common + [
            (['model', 'type'], model_type),
            (['model', 'fusion_type'], 'linear'),
            (['model', 'pool_type'], 'attention'),
        ], label)

    # ─── Tier 1.2: Pool variants (3) ───
    print("\n" + "─" * 60)
    print(" Tier 1.2: Pooling Ablation (3 experiments)")
    print("─" * 60)
    for pool in ['mean', 'max', 'cls']:
        run_step(config, f"pool={pool}", common + [
            (['model', 'type'], 'multichannel_encoder'),
            (['model', 'pool_type'], pool),
            (['model', 'fusion_type'], 'linear'),
        ], f"clean_pool_{pool}")

    # ─── Tier 1.3: Fusion gated (1) ───
    print("\n" + "─" * 60)
    print(" Tier 1.3: Fusion Ablation (1 experiment)")
    print("─" * 60)
    run_step(config, "fusion=gated", common + [
        (['model', 'type'], 'multichannel_encoder'),
        (['model', 'fusion_type'], 'gated'),
        (['model', 'pool_type'], 'attention'),
    ], "clean_fusion_gated")

    # ─── Tier 1.4: Combined best (1) — gated fusion, attention pool, no contrastive ───
    print("\n" + "─" * 60)
    print(" Tier 1.4: Combined Best (no contrastive) (1 experiment)")
    print("─" * 60)
    run_step(config, "combined_best", common + [
        (['model', 'type'], 'multichannel_encoder'),
        (['model', 'fusion_type'], 'gated'),
        (['model', 'pool_type'], 'attention'),
    ], "clean_combined_best")

    # ─── Tier 2a: Contrastive variants @ γ=0.3 (3) ───
    print("\n" + "─" * 60)
    print(" Tier 2a: Contrastive Variants (3 experiments, γ=0.3)")
    print("─" * 60)
    variant_results = []
    for variant in ['nat', 'lang', 'hier']:
        metrics = run_step(config, f"contrast={variant}", common + [
            (['model', 'type'], 'multichannel_encoder'),
            (['model', 'fusion_type'], 'linear'),
            (['model', 'pool_type'], 'attention'),
            (['train', 'contrastive_weight'], 0.3),
            (['train', 'contrastive_variant'], variant),
        ], f"clean_contrast_{variant}")
        if metrics is not None:
            metrics['_variant'] = variant
        variant_results.append(metrics)

    # 최적 variant 선택 (nat_top1 기준)
    valid = [r for r in variant_results if r is not None]
    if valid:
        best_v = max(valid, key=lambda r: r.get('nat_top1', -1))
        best_variant = best_v['_variant']
        print(f"\n  ★ Best contrastive variant = '{best_variant}' "
              f"(nat_top1={best_v['nat_top1']:.2f}, "
              f"ρ_all={best_v.get('cnl_rho_all', 0):.4f})")
    else:
        best_variant = 'hier'
        print(f"\n  [WARN] All variants failed; falling back to '{best_variant}'")

    # ─── Tier 2b: γ sweep with best variant (2) ───
    print("\n" + "─" * 60)
    print(f" Tier 2b: γ Sweep on '{best_variant}' (2 experiments)")
    print("─" * 60)
    for gamma in [0.1, 0.5]:
        run_step(config, f"γ={gamma} variant={best_variant}", common + [
            (['model', 'type'], 'multichannel_encoder'),
            (['model', 'fusion_type'], 'linear'),
            (['model', 'pool_type'], 'attention'),
            (['train', 'contrastive_weight'], gamma),
            (['train', 'contrastive_variant'], best_variant),
        ], f"clean_contrast_{best_variant}_g{gamma}")

    # ─── Tier 3: Data ablation (2) ───
    print("\n" + "─" * 60)
    print(" Tier 3: Synthetic Data Ablation (2 experiments)")
    print("─" * 60)
    for syn_target, label_suffix in [(400, 't400'), (1000, 't1000')]:
        run_step(config, f"syn @ target={syn_target}", common + [
            (['model', 'type'], 'multichannel_encoder'),
            (['model', 'fusion_type'], 'linear'),
            (['model', 'pool_type'], 'attention'),
            (['data', 'synthetic_path'], './data/raw/synthetic_names.csv'),
            (['data', 'synthetic_target'], syn_target),
        ], f"clean_syn_{label_suffix}")

    print("\n" + "*" * 60)
    print(" Clean Rerun 완료 (16 experiments)")
    print(f"   Best contrastive variant detected: '{best_variant}'")
    print("*" * 60)


# ─── Multi-Channel Conformer (MCC) — main Architecture novelty ───

def step_mcc(config):
    """
    Multi-Channel Conformer 검증 실험.

    설계 (3 design choices):
      1. 채널별 Conformer stack (uni/bi/tri 각자 specialized)
      2. 채널별 다른 conv kernel size (uni→2, bi→3, tri→5)
      3. Cross-channel attention으로 fusion

    페이퍼 contribution:
      · Architecture novelty (단계 3, combination + task-specific design)
      · Conformer (음성)을 multichannel character NLP에 task-specific 재설계
      · Self-attention 보존 → Phase 4 trigram 분석 그대로 가능
    """
    print("\n" + "=" * 60)
    print(" Multi-Channel Conformer (MCC)")
    print("=" * 60)

    common = [
        (['data', 'augment'], False),
        (['train', 'class_weight'], False),
        (['data', 'synthetic_path'], None),
        (['train', 'alpha'], 0.5),
        (['train', 'learning_rate'], 5e-5),
        (['model', 'n_layers'], 4),
        (['model', 'embedding_dim'], 256),
        (['model', 'hidden_dim'], 1024),
        (['model', 'n_head'], 8),
        (['data', 'max_length'], 90),
        (['data', 'ngram'], 'multi'),
        (['train', 'contrastive_weight'], 0.0),
        (['model', 'pool_type'], 'attention'),
        # ─── 핵심: MCC ───
        (['model', 'type'], 'multichannel_conformer'),
        (['model', 'conv_kernels'], [2, 3, 5]),   # uni / bi / tri
    ]

    run_step(config, "MCC (kernels=[2,3,5])", common, "clean_mcc")

    print("\n" + "*" * 60)
    print(" Multi-Channel Conformer 실험 완료")
    print("*" * 60)


# ─── MCC Tune: cumulative 4-experiment overnight push ───

def step_mcc_tune(config):
    """
    MCC 성능 극대화 — 누적 4개 실험.

    각 EXP는 직전 EXP의 변경을 그대로 carry하면서 한 가지를 추가:
      EXP1: optimization (LR↑ + warmup/cosine + dropout↓)
      EXP2: + tactical (multi-kernel conv [3,5,7] + label smoothing)
      EXP3: + architecture (cross-attn per layer — ρ 회복 가설)
      EXP4: + capacity (n_layers 4 → 6)

    각 EXP ~3~4h, 총 약 13.5h. 익일 출근 시 EXP4 = 최종 best 후보.

    NOTE: cumulative 디자인은 paper의 ablation 깔끔함은 떨어지지만,
    overnight max-performance push에는 최적. ablation은 추후 multi-seed에서.
    """
    print("\n" + "=" * 60)
    print(" MCC Tune (4 experiments, cumulative)")
    print("=" * 60)

    # 모든 EXP에 공통: MCC base + 정리된 학습 조건
    common = [
        (['data', 'augment'], False),
        (['train', 'class_weight'], False),
        (['data', 'synthetic_path'], None),
        (['train', 'alpha'], 0.5),
        (['model', 'embedding_dim'], 256),
        (['model', 'hidden_dim'], 1024),
        (['model', 'n_head'], 8),
        (['data', 'max_length'], 90),
        (['data', 'ngram'], 'multi'),
        (['train', 'contrastive_weight'], 0.0),
        (['model', 'pool_type'], 'attention'),
        (['model', 'type'], 'multichannel_conformer'),
        # baseline MCC (이전 실험과 동일)
        (['model', 'n_layers'], 4),
        (['model', 'conv_kernels'], [2, 3, 5]),
        (['model', 'cross_attn_per_layer'], False),
        # baseline trainer 옵션 (각 EXP에서 override)
        (['train', 'learning_rate'], 5e-5),
        (['train', 'lr_schedule'], 'fixed'),
        (['model', 'dropout'], 0.2),
        (['train', 'label_smoothing'], 0.0),
    ]

    # ─── EXP1: Optimization tuning ───
    # Conformer 표준에 맞춘 LR + schedule + dropout. 가장 큰 단일 레버.
    print("\n--- EXP1: LR 2e-4 + warmup/cosine + dropout 0.1 ---")
    exp1 = common + [
        (['train', 'learning_rate'], 2e-4),
        (['train', 'lr_schedule'], 'cosine_warmup'),
        (['train', 'warmup_ratio'], 0.05),
        (['model', 'dropout'], 0.1),
    ]
    run_step(config, "MCC Tune EXP1: opt tuning", exp1, "mcc_tune_exp1_opt")

    # ─── EXP2: + multi-kernel conv + label smoothing ───
    # CNN 강점 차용 (multi-scale local) + 168-class CE calibration
    print("\n--- EXP2: + conv_kernels [3,5,7] + label_smoothing 0.1 ---")
    exp2 = exp1 + [
        (['model', 'conv_kernels'], [3, 5, 7]),
        (['train', 'label_smoothing'], 0.1),
    ]
    run_step(config, "MCC Tune EXP2: + kernels + smoothing", exp2, "mcc_tune_exp2_tactical")

    # ─── EXP3: + cross-attn per layer ───
    # Channel-fragmentation 완화 → ρ 회복 가설 직접 테스트
    print("\n--- EXP3: + cross_attn_per_layer ---")
    exp3 = exp2 + [
        (['model', 'cross_attn_per_layer'], True),
    ]
    run_step(config, "MCC Tune EXP3: + cross-attn per layer", exp3, "mcc_tune_exp3_xattn")

    # ─── EXP4: + n_layers 6 ───
    # Capacity push (위에서 underfit이라 가정 시 효과 클 것)
    print("\n--- EXP4: + n_layers 6 ---")
    exp4 = exp3 + [
        (['model', 'n_layers'], 6),
    ]
    run_step(config, "MCC Tune EXP4: + n_layers 6", exp4, "mcc_tune_exp4_layers6")

    print("\n" + "*" * 60)
    print(" MCC Tune 완료 (4 experiments)")
    print("   결과 비교: experiment_results.csv 확인")
    print("   기준: nat_top1, lang_top1, ρ_all 함께 보기 (Pareto)")
    print("*" * 60)


# ─── MCC Single Combo: LR 1e-4 + dropout 0.5 + batch 256 ───

def step_mcc_combo(config):
    """
    단일 시도: 어제 mcc_tune 4-실험이 cumulative LR↑/dropout↓로 모두 손해본 것의
    정반대 방향 — LR 적당히 ↑ + dropout 강 reg + batch ↑.

    가설:
      · MCC가 head class에서 약간 overfit (top1 48.98 vs F1 33.71 gap)
      · 강한 dropout(0.5)이 minor class macro F1에 도움
      · batch 256 → gradient 노이즈 ↓ → high dropout과 시너지
      · LR 1e-4는 batch 2x에 맞춘 linear scaling (baseline 5e-5의 2x)

    예상:
      · nat_top1: 48.5~49.7 (50/30/20 = noise/win/lose)
      · macro_f1: 33~36 (regularization → minor class 살리기)
      · ρ_all:  0.36~0.39
    """
    print("\n" + "=" * 60)
    print(" MCC Single Combo (LR 1e-4 + dropout 0.5 + batch 256)")
    print("=" * 60)

    overrides = [
        # ─── baseline MCC와 동일 (변경 없음) ───
        (['data', 'augment'], False),
        (['train', 'class_weight'], False),
        (['data', 'synthetic_path'], None),
        (['train', 'alpha'], 0.5),
        (['model', 'embedding_dim'], 256),
        (['model', 'hidden_dim'], 1024),
        (['model', 'n_head'], 8),
        (['data', 'max_length'], 90),
        (['data', 'ngram'], 'multi'),
        (['train', 'contrastive_weight'], 0.0),
        (['model', 'pool_type'], 'attention'),
        (['model', 'type'], 'multichannel_conformer'),
        (['model', 'n_layers'], 4),
        (['model', 'conv_kernels'], [2, 3, 5]),
        (['model', 'cross_attn_per_layer'], False),
        (['train', 'lr_schedule'], 'fixed'),       # 어제 cosine_warmup 실패 → fixed로 회귀
        (['train', 'label_smoothing'], 0.0),       # 어제 label smoothing test_loss 왜곡 → 끔
        # ─── 핵심 변경: LR + dropout + batch ───
        (['train', 'learning_rate'], 1e-4),
        (['model', 'dropout'], 0.5),
        (['train', 'batch_size'], 256),
    ]

    run_step(config, "MCC Combo (LR 1e-4 + drop 0.5 + bs 256)",
             overrides, "mcc_combo_lr1e4_drop05_bs256")

    print("\n" + "*" * 60)
    print(" MCC Combo 실험 완료")
    print("   비교 anchor:")
    print("     baseline MCC: nat_top1 48.98, F1 33.71, ρ_all 0.368")
    print("*" * 60)


# ─── MCC + Conv Stem: borrow CNN inductive bias before attention ───

def step_mcc_convstem(config):
    """
    MCC + Conv Stem 실험.

    가설:
      · CNN이 50.16%로 MCC(48.98) 1.18pp 앞선 이유 = local pattern detection의
        강한 inductive bias (translation invariance + locality)
      · 이를 attention 전에 명시적 주입하면 self-attention은 그 위에서
        global aggregation에 집중 가능
      · ConvNeXt/ConvViT의 motivation을 character-level로

    설계:
      · Embedding → ConvStem (depthwise sep × 2, kernel 5) → PE → Conformer × 4
      · 채널별 독립 stem (uni/bi/tri 각자 학습)
      · Sequence length 보존 → cross-attn / pooling 영향 없음

    예상 (3 시나리오):
      · 50% 확률: nat_top1 49.0~49.5 (marginal, ConformerConv와 redundant)
      · 35% 확률: nat_top1 50.0+ (CNN inductive bias 주입 성공)
      · 15% 확률: 48.5 이하 (depth ↑로 underfit)
    """
    print("\n" + "=" * 60)
    print(" MCC + Conv Stem (CNN inductive bias 주입)")
    print("=" * 60)

    # ─── 변형 1: Conv stem만 (다른 옵션은 baseline MCC와 동일) ───
    common = [
        (['data', 'augment'], False),
        (['train', 'class_weight'], False),
        (['data', 'synthetic_path'], None),
        (['train', 'alpha'], 0.5),
        (['train', 'learning_rate'], 5e-5),
        (['train', 'batch_size'], 128),
        (['model', 'embedding_dim'], 256),
        (['model', 'hidden_dim'], 1024),
        (['model', 'n_head'], 8),
        (['model', 'n_layers'], 4),
        (['model', 'dropout'], 0.2),
        (['data', 'max_length'], 90),
        (['data', 'ngram'], 'multi'),
        (['train', 'contrastive_weight'], 0.0),
        (['train', 'lr_schedule'], 'fixed'),
        (['train', 'label_smoothing'], 0.0),
        (['model', 'pool_type'], 'attention'),
        (['model', 'type'], 'multichannel_conformer'),
        (['model', 'conv_kernels'], [2, 3, 5]),
        (['model', 'cross_attn_per_layer'], False),
        # ─── 핵심: conv stem ON ───
        (['model', 'conv_stem'], True),
        (['model', 'conv_stem_kernel'], 5),
        (['model', 'conv_stem_layers'], 2),
    ]

    print("\n--- Variant 1: ConvStem (kernel=5, layers=2) ---")
    run_step(config, "MCC + ConvStem (k=5, n=2)",
             common, "mcc_convstem_k5_n2")

    # ─── 변형 2: 더 큰 kernel (kernel=7) — 더 넓은 local context ───
    print("\n--- Variant 2: ConvStem kernel=7 ---")
    run_step(config, "MCC + ConvStem (k=7, n=2)",
             common + [(['model', 'conv_stem_kernel'], 7)],
             "mcc_convstem_k7_n2")

    print("\n" + "*" * 60)
    print(" MCC + ConvStem 실험 완료 (2 variants)")
    print("   비교 anchor:")
    print("     baseline MCC: nat_top1 48.98, F1 33.71, ρ_all 0.368")
    print("     CNN baseline: nat_top1 50.16, F1 36.73, ρ_all 0.321")
    print("   기준: nat_top1 50.0 넘으면 architecture novelty 강화")
    print("*" * 60)


# ─── Architecture Overnight: 4 distinct structural variants ───

def step_mcc_arch_overnight(config):
    """
    구조 변경 4종 — overnight 한 번에 비교.

    공통: baseline MCC 학습 조건 (lr 5e-5, dropout 0.2, batch 128, n_layers 4).
    차이는 architecture 한 변수만.

    EXP1: ConvStem standard (k=5, layers=2)        — baseline injection
    EXP2: ConvStem standard deep (k=5, layers=4)   — depth ablation
    EXP3: ConvStem multiscale ([3,5,7] parallel)   — Inception-style
    EXP4: Channel Ensemble Head (no cross-attn)    — output fusion strategy

    EXP1~3는 input-side bias 강화 (conv-first), EXP4는 output-side fusion 변경.
    네 가지가 각각 *독립적인* architectural intervention이라 paper ablation 표로 직접 사용 가능.
    """
    print("\n" + "=" * 60)
    print(" Architecture Overnight (4 variants)")
    print("=" * 60)

    # 공통 — baseline MCC 학습 조건
    common = [
        (['data', 'augment'], False),
        (['train', 'class_weight'], False),
        (['data', 'synthetic_path'], None),
        (['train', 'alpha'], 0.5),
        (['train', 'learning_rate'], 5e-5),
        (['train', 'batch_size'], 128),
        (['model', 'embedding_dim'], 256),
        (['model', 'hidden_dim'], 1024),
        (['model', 'n_head'], 8),
        (['model', 'n_layers'], 4),
        (['model', 'dropout'], 0.2),
        (['data', 'max_length'], 90),
        (['data', 'ngram'], 'multi'),
        (['train', 'contrastive_weight'], 0.0),
        (['train', 'lr_schedule'], 'fixed'),
        (['train', 'label_smoothing'], 0.0),
        (['model', 'pool_type'], 'attention'),
        (['model', 'type'], 'multichannel_conformer'),
        (['model', 'conv_kernels'], [2, 3, 5]),
        (['model', 'cross_attn_per_layer'], False),
        # 기본은 stem off, ensemble off — 각 EXP에서 override
        (['model', 'conv_stem'], False),
        (['model', 'channel_ensemble_head'], False),
    ]

    # ─── EXP1: ConvStem standard (k=5, layers=2) ───
    print("\n--- EXP1: ConvStem standard (k=5, layers=2) ---")
    run_step(config, "Arch EXP1: ConvStem k=5 n=2",
             common + [
                 (['model', 'conv_stem'], True),
                 (['model', 'conv_stem_type'], 'standard'),
                 (['model', 'conv_stem_kernel'], 5),
                 (['model', 'conv_stem_layers'], 2),
             ], "arch_exp1_convstem_k5_n2")

    # ─── EXP2: ConvStem standard deep (k=5, layers=4) ───
    print("\n--- EXP2: ConvStem deep (k=5, layers=4) ---")
    run_step(config, "Arch EXP2: ConvStem k=5 n=4",
             common + [
                 (['model', 'conv_stem'], True),
                 (['model', 'conv_stem_type'], 'standard'),
                 (['model', 'conv_stem_kernel'], 5),
                 (['model', 'conv_stem_layers'], 4),
             ], "arch_exp2_convstem_k5_n4")

    # ─── EXP3: Multi-scale ConvStem ([3,5,7] parallel) ───
    print("\n--- EXP3: Multi-scale ConvStem ([3,5,7]) ---")
    run_step(config, "Arch EXP3: ConvStem multiscale [3,5,7]",
             common + [
                 (['model', 'conv_stem'], True),
                 (['model', 'conv_stem_type'], 'multiscale'),
                 (['model', 'conv_stem_kernels'], [3, 5, 7]),
                 (['model', 'conv_stem_layers'], 2),
             ], "arch_exp3_convstem_multiscale")

    # ─── EXP4: Channel Ensemble Head ───
    print("\n--- EXP4: Channel Ensemble Head (no cross-attn fusion) ---")
    run_step(config, "Arch EXP4: Channel Ensemble Head",
             common + [
                 (['model', 'channel_ensemble_head'], True),
             ], "arch_exp4_channel_ensemble")

    print("\n" + "*" * 60)
    print(" Architecture Overnight 완료 (4 variants)")
    print("   비교 anchor:")
    print("     baseline MCC: nat_top1 48.98, F1 33.71, ρ_all 0.368")
    print("     CNN baseline: nat_top1 50.16, F1 36.73, ρ_all 0.321")
    print("   판단:")
    print("     · EXP 중 best가 50.0+ → architecture novelty 강화, paper main")
    print("     · 모두 49.5 미만 → 'architecture sufficient' 결론, baseline MCC 유지")
    print("*" * 60)


# ─── Multi-seed verification — paper main result 신뢰구간 확보 ───

def step_multi_seed(config):
    """
    Multi-seed 검증 — best 모델 + 핵심 baseline을 추가 시드로 재실행.

    배경:
      · seed=42 결과 (이미 존재):
          Ours (conf_alpha_0.3): nat 50.34 / F1 36.48 / lang 68.95 / ρ 0.401
          Ours+contrast (γ=0.3): nat 50.53 / F1 36.95 / ρ 0.382
          CNN baseline:           nat 50.16 / F1 36.73 / lang 68.65 / ρ 0.321
      · paper에 "50.34 ± σ" 형식 보고하려면 추가 2 seed 필요
      · 리뷰어 "single-seed, +0.18pp는 노이즈" critique 차단

    실행:
      seeds = [7, 123]  (seed 42는 기존 결과 사용)
      모델 3종 × 2 seed = 6 runs (~15h)

    우선순위 (compute 부족 시 위에서부터):
      1. Ours (α=0.3) — paper main "Ours" — 절대 필수
      2. CNN baseline — direct comparison anchor — 필수
      3. Ours + Contrast (γ=0.3) — secondary best — 가능하면

    각 모델은 각자의 best config 사용 (Ours: α=0.3 lr=3e-5, CNN: α=0.5 lr=5e-5).
    """
    print("\n" + "=" * 60)
    print(" Multi-Seed Verification (3 models × 3 seeds = 9 runs)")
    print("=" * 60)

    seeds = [42, 7, 123]   # 39-group 재학습으로 seed 42도 새로 필요

    # Ours config (conformer_overrides() 헬퍼 사용 — 기존 conf_alpha_0.3과 정확히 일치)
    # · channel_ensemble_head=True (★ FINAL choice from prior experiments)
    # · channel_fusion_type='uniform'
    # · α=0.3, lr=3e-5, n_layers=4, d_model=256  (grid best, 260513)
    def ours_overrides(seed, with_contrast=False):
        kwargs = dict(
            lr=3e-5, n_layers=4, d_model=256, alpha=0.3,
            channel_ensemble_head=True,
            channel_fusion_type='uniform',
            seed=seed,
        )
        if with_contrast:
            kwargs['contrastive_weight'] = 0.3
            kwargs['contrastive_variant'] = 'hier'
        return conformer_overrides(**kwargs)

    # CNN baseline — Position B: Ours와 동일 protocol (α=0.3, lr=3e-5, L=4)
    def cnn_overrides(seed):
        return [
            (['data', 'augment'], False),
            (['train', 'class_weight'], False),
            (['data', 'synthetic_path'], None),
            (['train', 'alpha'], 0.3),                    # Ours와 동일
            (['train', 'learning_rate'], 3e-5),           # Ours와 동일
            (['train', 'batch_size'], 128),
            (['model', 'embedding_dim'], 256),
            (['model', 'hidden_dim'], 1024),
            (['model', 'n_head'], 8),
            (['model', 'n_layers'], 4),                   # Ours와 동일
            (['data', 'max_length'], 90),
            (['data', 'ngram'], 'multi'),
            (['train', 'contrastive_weight'], 0.0),
            (['model', 'type'], 'multichannel_cnn'),
            (['model', 'fusion_type'], 'linear'),
            (['model', 'pool_type'], 'attention'),
            (['experiment', 'seed'], seed),
        ]

    # ─── 우선순위 순서로 실행 (interrupt 시 핵심 비교는 보존됨) ───
    # Round 0: LR baseline (sklearn, ~10min, deterministic — 1회만)
    # Round 1: Ours + CNN @ seed=7   — direct comparison 1조
    # Round 2: Ours + CNN @ seed=123 — direct comparison 2조 → mean ± std 가능
    # Round 3: Ours+Contrast @ both seeds — secondary
    #
    # skip_seeds: 이미 동일 protocol·동일 seed로 학습된 .pt가 있어 재학습 불필요한 seed.
    #   · Ours seed=42       → 260514/conf_alpha_0.3.pt 재사용 (step_conformer_alpha와 동일 헬퍼)
    #   · CNN seed=42        → 260513/baseline_cnn.pt 재사용 (Position B α=0.3 lr=3e-5 L=4)
    #   · +Contrast seed=42  → 38-group 시절 결과만 있어 39-group 재학습 필요 → skip 없음
    rounds = [
        # (label, factory, save_prefix_template, factory_kwargs, skip_seeds)
        ("Ours (α=0.3)",            ours_overrides, "multiseed_ours_alpha03_seed{}", {}, [42]),
        ("CNN baseline",             cnn_overrides,  "multiseed_baseline_cnn_seed{}", {}, [42]),
        ("Ours + Contrast (γ=0.3)",  ours_overrides, "multiseed_ours_contrast_seed{}", {'with_contrast': True}, []),
    ]

    # ─── LR baseline은 multi_seed에서 제외 ───
    # deterministic 1회 baseline이라 별도 step (`--step lr_baseline`)으로 실행.

    for seed in seeds:
        print(f"\n{'━' * 60}")
        print(f"  ROUND seed={seed}")
        print(f"{'━' * 60}")
        for label, factory, prefix_tmpl, fkwargs, skip_seeds in rounds:
            if seed in skip_seeds:
                print(f"\n  [SKIP] {label} seed={seed} — 기존 결과 재사용")
                continue
            run_step(
                config,
                f"{label} seed={seed}",
                factory(seed=seed, **fkwargs),
                prefix_tmpl.format(seed),
            )

    print("\n" + "*" * 60)
    print(" Multi-Seed Verification 완료")
    print("   결과 분석:")
    print("     experiment_results.csv에서 'multiseed_*' 행 추출 →")
    print("     기존 seed=42 결과와 결합하여 mean ± std 계산")
    print("   페이퍼 표:")
    print("     'Model: nat_top1 (mean ± std over 3 seeds)'")
    print("*" * 60)


# ─── Multi-Seed for Remaining Table 1/2 Rows ───

def step_multi_seed_extras(config):
    """
    Multi-seed verification for the remaining Table 1 baselines and
    Table 2 ablation rows that are still single-seed reference points.

    배경:
      · step_multi_seed는 Ours / CNN / +Contrast 3 모델만 다룸 (Table 1 두 행 + Table 2 (a)(f))
      · Table 1의 나머지 baseline (FastText, BiLSTM, Transformer, Vanilla MCC)와
        Table 2의 ablation 행 (α=0.0 / 0.5 / 0.7, Gate fusion)은 single-seed.
      · reviewer pushback 차단을 위해 동일 protocol(seeds {42, 7, 123})로 확장.

    실행:
      seeds = [42, 7, 123]
      configs 8종 × 3 seeds = 24 runs (~50-70h, 전체 분량은 다일 야간 분할 권장)

      · Vanilla MCC = Table 1 row 6 = Table 2 row (d) — 한 번만 학습해서 두 표 공유
      · α=0.3 Ours, +Contrast는 step_multi_seed에서 이미 끝났으므로 제외

    Position B protocol (전 모델 공통):
      α=0.3 (단, α-ablation row는 0.0 / 0.5 / 0.7), lr=3e-5, n_layers=4, d_model=256
      augment=False, class_weight=False (single-seed reference와 동일 framework)
    """
    print("\n" + "=" * 60)
    print(" Multi-Seed for Remaining Table 1 / Table 2 Rows")
    print(" 8 configs × 3 seeds = 24 runs (~50-70h)")
    print("=" * 60)

    seeds = [42, 7, 123]

    # ─── Position B baseline overrides (단순 모델: FT / BiLSTM / Transformer) ───
    def simple_baseline_overrides(model_type, seed):
        return [
            (['train', 'alpha'], 0.3),
            (['train', 'learning_rate'], 3e-5),
            (['train', 'class_weight'], False),
            (['train', 'contrastive_weight'], 0.0),
            (['data', 'augment'], False),
            (['data', 'synthetic_path'], None),
            (['data', 'ngram'], 'multi'),
            (['data', 'max_length'], 90),
            (['model', 'type'], model_type),
            (['model', 'n_layers'], 4),
            (['model', 'embedding_dim'], 256),
            (['model', 'hidden_dim'], 1024),
            (['model', 'n_head'], 8),
            (['model', 'fusion_type'], 'linear'),
            (['model', 'pool_type'], 'attention'),
            (['experiment', 'seed'], seed),
        ]

    # ─── Conformer-family overrides (Vanilla MCC + α-ablation + Gate) ───
    def conf(seed, alpha=0.3,
             channel_ensemble_head=True,
             channel_fusion_type='uniform'):
        return conformer_overrides(
            lr=3e-5, n_layers=4, d_model=256, alpha=alpha,
            channel_ensemble_head=channel_ensemble_head,
            channel_fusion_type=channel_fusion_type,
            seed=seed,
        )

    # (label, factory, save_prefix_template)
    rounds = [
        # ─── Table 1 baselines (Position B, 1-seed → 3-seed) ───
        ("FastText",
         lambda seed: simple_baseline_overrides('multichannel_fasttext', seed),
         "multiseed_baseline_fasttext_seed{}"),
        ("BiLSTM",
         lambda seed: simple_baseline_overrides('multichannel_bilstm', seed),
         "multiseed_baseline_bilstm_seed{}"),
        ("Transformer",
         lambda seed: simple_baseline_overrides('multichannel_encoder', seed),
         "multiseed_baseline_transformer_seed{}"),

        # ─── Vanilla MCC: Table 1 row 6 = Table 2 row (d) (CEH OFF) ───
        ("Vanilla MCC (no CEH)",
         lambda seed: conf(seed, channel_ensemble_head=False),
         "multiseed_vanilla_mcc_seed{}"),

        # ─── Table 2 α-ablation (a' / b / c): Ours arch + alpha 변경 ───
        ("α=0.0 (no lang head)",
         lambda seed: conf(seed, alpha=0.0),
         "multiseed_alpha_0.0_seed{}"),
        ("α=0.5",
         lambda seed: conf(seed, alpha=0.5),
         "multiseed_alpha_0.5_seed{}"),
        ("α=0.7",
         lambda seed: conf(seed, alpha=0.7),
         "multiseed_alpha_0.7_seed{}"),

        # ─── Table 2 (e) Gate fusion (CEH ON + gate) ───
        ("CEH + Gate fusion",
         lambda seed: conf(seed, channel_fusion_type='gate'),
         "multiseed_gate_fusion_seed{}"),
    ]

    for seed in seeds:
        print(f"\n{'━' * 60}")
        print(f"  ROUND seed={seed}")
        print(f"{'━' * 60}")
        for label, factory, prefix_tmpl in rounds:
            run_step(
                config,
                f"{label} seed={seed}",
                factory(seed),
                prefix_tmpl.format(seed),
            )

    print("\n" + "*" * 60)
    print(" Multi-Seed Extras 완료")
    print("   결과 분석:")
    print("     experiment_results.csv에서 다음 prefix 추출:")
    print("       multiseed_baseline_fasttext_seed*")
    print("       multiseed_baseline_bilstm_seed*")
    print("       multiseed_baseline_transformer_seed*")
    print("       multiseed_vanilla_mcc_seed*")
    print("       multiseed_alpha_0.0_seed* / 0.5 / 0.7")
    print("       multiseed_gate_fusion_seed*")
    print("     → mean ± std 계산 후 Table 1 / Table 2 placeholder 갱신")
    print("*" * 60)


# ─── Linguistic-Aware Fusion (Architecture novelty) ───

def step_fusion_linguistic(config):
    """
    Linguistic-aware fusion 검증 실험 — 2 variants.

    아이디어 (Architecture novelty):
      uni/bi/tri channel weights를 언어권별로 다르게 학습.
      slavic → tri 우세 (-ova), 영미 짧은 이름 → uni 균형 등.

    실험 구성 (sequential dependency):
      1. uniform init  (lang_prior = zeros)
         → "language-conditioned fusion" 자체 효과 측정
      2. typology init (data-derived empirical prior로 초기화)
         → "linguistic prior 주입" 효과 측정 (Vaswani-style structured init)

    typology JSON은 미리 생성되어 있어야 함:
      python -m analysis.compute_typology_prior \\
          --output data/linguistics/typology_prior.json
    """
    print("\n" + "=" * 60)
    print(" Linguistic-Aware Fusion (2 experiments, Architecture novelty)")
    print("=" * 60)

    common = [
        (['data', 'augment'], False),
        (['train', 'class_weight'], False),
        (['train', 'alpha'], 0.5),
        (['train', 'learning_rate'], 5e-5),
        (['model', 'n_layers'], 4),
        (['model', 'embedding_dim'], 256),
        (['model', 'hidden_dim'], 1024),
        (['model', 'n_head'], 8),
        (['data', 'max_length'], 90),
        (['data', 'ngram'], 'multi'),
        (['train', 'contrastive_weight'], 0.0),
        (['model', 'type'], 'multichannel_encoder'),
        (['model', 'pool_type'], 'attention'),
        (['model', 'fusion_type'], 'linguistic'),
        (['train', 'aux_lang_weight'], 0.1),
    ]

    # 1. uniform init
    print("\n--- Variant 1: uniform init (data-driven only) ---")
    run_step(config, "fusion=linguistic (uniform init)", common + [
        (['model', 'typology_path'], 'NONEXISTENT'),  # 파일 없음 → uniform fallback
    ], "clean_fusion_linguistic_uniform")

    # 2. typology init (Vaswani-style structured prior)
    print("\n--- Variant 2: typology init (empirical linguistic prior) ---")
    run_step(config, "fusion=linguistic (typology init)", common + [
        (['model', 'typology_path'], 'data/linguistics/typology_prior.json'),
    ], "clean_fusion_linguistic_typology")

    print("\n" + "*" * 60)
    print(" Linguistic fusion (2 variants) 실험 완료")
    print("*" * 60)


# ─── Combo Probe: syn@400 + fusion=gated ───

def step_syn_gated_combo(config):
    """
    Top-2 ρ improvers (syn @ 400, fusion=gated)을 합친 단일 검증 실험.

    가설: 둘 다 ρ_all을 미세하게 올렸으니 합쳐도 추가 이득 있는가?
      · syn @ 400 단독:    ρ_all = 0.3953
      · fusion=gated 단독: ρ_all = 0.3889
      · default:            ρ_all = 0.3792

    예상: marginal (0.39~0.41 사이) — single-seed 노이즈 범위 가능성 높음
    실패 시 (≤0.38): "individual gains do not compose" finding
    성공 시 (≥0.41): "combined synthetic + gated yields best ρ"
    """
    print("\n" + "=" * 60)
    print(" Combo Probe: syn @ 400 + fusion=gated (1 experiment)")
    print("=" * 60)

    common = [
        (['data', 'augment'], False),
        (['train', 'class_weight'], False),
        (['train', 'alpha'], 0.5),
        (['train', 'learning_rate'], 5e-5),
        (['model', 'n_layers'], 4),
        (['model', 'embedding_dim'], 256),
        (['model', 'hidden_dim'], 1024),
        (['model', 'n_head'], 8),
        (['data', 'max_length'], 90),
        (['data', 'ngram'], 'multi'),
        (['train', 'contrastive_weight'], 0.0),
        (['model', 'type'], 'multichannel_encoder'),
        (['model', 'pool_type'], 'attention'),
        # ─── 핵심 변경: 두 best ρ improver 결합 ───
        (['model', 'fusion_type'], 'gated'),
        (['data', 'synthetic_path'], './data/raw/synthetic_names.csv'),
        (['data', 'synthetic_target'], 400),
    ]

    run_step(config, "syn@400 + gated", common, "clean_syn400_gated")

    print("\n" + "*" * 60)
    print(" Combo probe 완료")
    print("*" * 60)


# ════════════════════════════════════════════════════════════════════
#  Conformer + Channel Ensemble + Gate 전용 step (2026-04-30 추가)
#  · weekend_full pipeline에서 사용. 기존 TE 기준 step과 분리.
# ════════════════════════════════════════════════════════════════════

# 모든 Conformer 실험의 공통 설정 (base_config.yaml의 multichannel_conformer 매핑)
# arch_exp4 (FINAL) = ConvStem off + Channel Ensemble + Uniform fusion + ConvKernels(2,3,5)
# Gate fusion 시도했으나 -2.3pp Nat-T1 하락으로 negative finding 처리
CONFORMER_DEFAULTS = {
    'model_type': 'multichannel_conformer',
    'ngram': 'multi',
    'pool_type': 'attention',
    'fusion_type': 'linear',
    'conv_kernels': [2, 3, 5],
    'conv_stem': False,                 # exp4 단독이 가장 좋음 (결합 실패 검증됨)
    'channel_ensemble_head': True,      # exp4 핵심
    'channel_fusion_type': 'uniform',   # ★ FINAL: 1/3 평균 (gate ablation 결과 열등)
    'augment': False,
    'class_weight': False,
    'max_length': 90,
}


def conformer_overrides(lr, n_layers, d_model, alpha,
                        ngram='multi',
                        conv_kernels=(2, 3, 5),
                        channel_ensemble_head=True,
                        channel_fusion_type='uniform',   # ★ default uniform (FINAL)
                        contrastive_weight=0.0,
                        contrastive_variant='hier',
                        seed=42,
                        max_length=90):
    """Conformer + Gate 실험 공통 override 빌더."""
    hidden_dim = d_model * 4
    n_head = 8 if d_model >= 256 else 4
    return [
        # Train
        (['train', 'learning_rate'], lr),
        (['train', 'alpha'], alpha),
        (['train', 'class_weight'], False),
        (['train', 'contrastive_weight'], contrastive_weight),
        (['train', 'contrastive_variant'], contrastive_variant),
        # Data
        (['data', 'ngram'], ngram),
        (['data', 'max_length'], max_length),
        (['data', 'augment'], False),
        # Model — Conformer 전용
        (['model', 'type'], 'multichannel_conformer'),
        (['model', 'n_layers'], n_layers),
        (['model', 'embedding_dim'], d_model),
        (['model', 'hidden_dim'], hidden_dim),
        (['model', 'n_head'], n_head),
        (['model', 'pool_type'], 'attention'),
        (['model', 'fusion_type'], 'linear'),
        (['model', 'conv_kernels'], list(conv_kernels)),
        (['model', 'conv_stem'], False),
        (['model', 'channel_ensemble_head'], channel_ensemble_head),
        (['model', 'channel_fusion_type'], channel_fusion_type),
        # Reproducibility
        (['experiment', 'seed'], seed),
    ]


# ─── Step W1: Conformer Grid Search (lr × layers, d_model=256 fix) ───

def step_conformer_grid(config):
    """
    arch_exp4 (FINAL: ConvStem off + Channel Ensemble + Uniform fusion) baseline에서
    lr × layers × d_model 그리드 탐색.
    · base_config.yaml의 grid_search 섹션 사용
    · best (lr, n_layers, d_model) 반환 → 후속 step들 baseline으로 사용
    """
    print("\n" + "=" * 60)
    print(" Step W1: arch_exp4 Grid Search (lr × layers × d_model)")
    print("        baseline = ConvStem off + Channel Ensemble + Uniform fusion")
    print("=" * 60)

    grid = config['grid_search']
    learning_rates = [float(x) for x in grid['learning_rates']]
    layers_list = grid.get('layers', [4])
    d_models = grid.get('d_models', [256])

    combinations = list(itertools.product(learning_rates, layers_list, d_models))
    print(f"  조합 수: {len(combinations)} (lr × layers × d_model)")

    results = []
    for i, (lr, n_layers, d_model) in enumerate(combinations, 1):
        prefix = f"conf_grid_lr{lr}_L{n_layers}_d{d_model}"
        print(f"\n  [{i}/{len(combinations)}] lr={lr}, layers={n_layers}, d_model={d_model}")

        metrics = run_step(config, f"Conf Grid {i}/{len(combinations)}",
                           conformer_overrides(lr, n_layers, d_model, alpha=0.5),
                           prefix)
        if metrics:
            metrics['_lr'] = lr
            metrics['_n_layers'] = n_layers
            metrics['_d_model'] = d_model
        results.append(metrics)

    best = select_best(results, 'nat_top1', higher_is_better=True)
    print_best("Conformer 최적 hparam", best,
               ['nat_top1', 'nat_macro_f1', 'lang_top1', 'cnl_rho_subset',
                '_lr', '_n_layers', '_d_model'])
    return {
        'learning_rate': best['_lr'],
        'n_layers': best['_n_layers'],
        'd_model': best['_d_model'],
    }


# ─── Step W2: α Ablation (arch_exp4 baseline) ───

def step_conformer_alpha(config, best_hp):
    """α 5-point ablation. arch_exp4 baseline + best_hp."""
    print("\n" + "=" * 60)
    print(" Step W2: Alpha Ablation (arch_exp4 baseline)")
    print(f"        (lr={best_hp['learning_rate']}, layers={best_hp['n_layers']}, d_model={best_hp['d_model']})")
    print("=" * 60)

    alphas = [0.0, 0.3, 0.5, 0.7]    # α=1.0 (lang-only) 제외
    results = []
    for alpha in alphas:
        prefix = f"conf_alpha_{alpha}"
        print(f"\n  α={alpha}")
        metrics = run_step(config, f"α={alpha}",
                           conformer_overrides(
                               best_hp['learning_rate'], best_hp['n_layers'],
                               best_hp['d_model'], alpha=alpha),
                           prefix)
        if metrics:
            metrics['_alpha'] = alpha
        results.append(metrics)

    # nat_top1 + lang_top1 합산 기준으로 best α 선택
    for r in results:
        if r is not None:
            r['_combined'] = r['nat_top1'] + r['lang_top1']
    best = select_best(results, '_combined', higher_is_better=True)
    print_best(f"최적 α = {best['_alpha']}", best,
               ['nat_top1', 'nat_macro_f1', 'lang_top1', 'cnl_rho_subset'])
    return best['_alpha']


# ─── Step W3: Channel Ablation (uni/bi/tri/multi) ───

def step_conformer_channel(config, best_hp, best_alpha):
    """채널별 ablation. Conformer는 multi 기본 — 단일 채널은 mc_encoder로 대체."""
    print("\n" + "=" * 60)
    print(f" Step W3: Channel Ablation (α={best_alpha})")
    print("=" * 60)

    # multi: Conformer + Gate (full)
    # uni/bi/tri: 단일 채널 — multichannel_conformer는 multi 전제이므로
    #             transformer_encoder로 단일 채널 baseline 대체 (TE 단일 채널)
    channel_specs = [
        ('uni',   'transformer_encoder'),
        ('bi',    'transformer_encoder'),
        ('tri',   'transformer_encoder'),
        ('multi', 'multichannel_conformer'),
    ]
    for ngram, model_type in channel_specs:
        prefix = f"conf_channel_{ngram}"
        if model_type == 'multichannel_conformer':
            overrides = conformer_overrides(
                best_hp['learning_rate'], best_hp['n_layers'],
                best_hp['d_model'], alpha=best_alpha, ngram='multi')
        else:
            # 단일 채널 TE
            d_model = best_hp['d_model']
            overrides = [
                (['train', 'learning_rate'], best_hp['learning_rate']),
                (['train', 'alpha'], best_alpha),
                (['train', 'class_weight'], False),
                (['data', 'ngram'], ngram),
                (['data', 'max_length'], 90),
                (['data', 'augment'], False),
                (['model', 'type'], model_type),
                (['model', 'n_layers'], best_hp['n_layers']),
                (['model', 'embedding_dim'], d_model),
                (['model', 'hidden_dim'], d_model * 4),
                (['model', 'n_head'], 8 if d_model >= 256 else 4),
                (['model', 'pool_type'], 'attention'),
            ]
        print(f"\n  channel={ngram} ({model_type})")
        run_step(config, f"channel={ngram}", overrides, prefix)


# ─── Step W4: Conv Kernels Ablation ───

def step_conformer_conv_kernels(config, best_hp, best_alpha):
    """채널별 conv kernel 조합 ablation (논문 main novelty 정당화)."""
    print("\n" + "=" * 60)
    print(f" Step W4: Conv Kernels Ablation (α={best_alpha})")
    print("=" * 60)

    kernel_specs = [
        ((2, 3, 5), 'k235'),     # 현재 default — channel-specific
        ((3, 3, 3), 'k333'),     # 동일 kernel
        ((5, 5, 5), 'k555'),     # 동일 큰 kernel
        ((3, 5, 7), 'k357'),     # 다른 조합
    ]
    for kernels, label in kernel_specs:
        prefix = f"conf_kernels_{label}"
        print(f"\n  kernels={kernels}")
        run_step(config, f"kernels={kernels}",
                 conformer_overrides(
                     best_hp['learning_rate'], best_hp['n_layers'],
                     best_hp['d_model'], alpha=best_alpha,
                     conv_kernels=kernels),
                 prefix)


# ─── Step W5: Channel Fusion Ablation (uniform baseline vs gate) ───

def step_conformer_fusion(config, best_hp, best_alpha):
    """
    Channel fusion ablation:
      · uniform (FINAL baseline): 1/3 평균
      · gate (negative finding): input-dependent MoE-style
    초기 실험에서 gate가 -2.3pp Nat-T1 하락 → uniform이 final.
    이 ablation은 gate가 왜 안 되는지 본문에서 정당화하는 데 사용.
    이미 단독 결과 보유 시 skip 가능 (가독성 위해 자동 진행).
    """
    print("\n" + "=" * 60)
    print(f" Step W5: Channel Fusion Ablation (α={best_alpha})")
    print("        baseline = uniform, ablation = gate (negative finding)")
    print("=" * 60)

    for fusion_type in ['uniform', 'gate']:
        prefix = f"conf_fusion_{fusion_type}"
        print(f"\n  channel_fusion_type={fusion_type}")
        run_step(config, f"fusion={fusion_type}",
                 conformer_overrides(
                     best_hp['learning_rate'], best_hp['n_layers'],
                     best_hp['d_model'], alpha=best_alpha,
                     channel_fusion_type=fusion_type),
                 prefix)


# ─── Step W6: Channel Ensemble Head Ablation ───

def step_conformer_ensemble(config, best_hp, best_alpha):
    """Channel Ensemble Head on/off."""
    print("\n" + "=" * 60)
    print(f" Step W6: Channel Ensemble Head Ablation (α={best_alpha})")
    print("=" * 60)

    # ON: 채널별 독립 head + uniform 1/3 평균 (arch_exp4 FINAL)
    # OFF: cross-attn으로 채널 fusion → single head (vanilla MCC ≈ mcc_combo)
    for head_on in [True, False]:
        label = 'ensemble_on' if head_on else 'ensemble_off'
        prefix = f"conf_{label}"
        print(f"\n  channel_ensemble_head={head_on}")
        # head ON: uniform 평균 (FINAL); head OFF: fusion_type은 무관 (single head)
        run_step(config, label,
                 conformer_overrides(
                     best_hp['learning_rate'], best_hp['n_layers'],
                     best_hp['d_model'], alpha=best_alpha,
                     channel_ensemble_head=head_on,
                     channel_fusion_type='uniform'),
                 prefix)


# ─── Step W7: Contrastive Auxiliary Loss Ablation ───

def step_conformer_contrastive(config, best_hp, best_alpha):
    """Contrastive auxiliary loss γ ablation (분석 강화 후보)."""
    print("\n" + "=" * 60)
    print(f" Step W7: Contrastive γ Ablation (α={best_alpha})")
    print("=" * 60)

    gammas = [0.3]   # γ=0.0은 fusion ablation의 'gate'와 동일
    for gamma in gammas:
        prefix = f"conf_contrast_g{gamma}_hier"
        print(f"\n  contrastive_weight={gamma}, variant=hier")
        run_step(config, f"γ={gamma}",
                 conformer_overrides(
                     best_hp['learning_rate'], best_hp['n_layers'],
                     best_hp['d_model'], alpha=best_alpha,
                     contrastive_weight=gamma,
                     contrastive_variant='hier'),
                 prefix)


# ─── Step W8: Multi-seed Ensemble (best config × N seeds) ───

def step_conformer_multi_seed(config, best_hp, best_alpha, seeds=(42, 123, 7)):
    """Best config로 여러 seed 학습 (variance 감소)."""
    print("\n" + "=" * 60)
    print(f" Step W8: Multi-seed Training (α={best_alpha}, seeds={seeds})")
    print("=" * 60)

    for seed in seeds:
        prefix = f"conf_seed{seed}"
        print(f"\n  seed={seed}")
        run_step(config, f"seed={seed}",
                 conformer_overrides(
                     best_hp['learning_rate'], best_hp['n_layers'],
                     best_hp['d_model'], alpha=best_alpha, seed=seed),
                 prefix)
    # 후속 작업: 별도 inference script로 logit 평균 ensemble


# ─── Step W9: DICL Analysis — Bootstrap CI + Paired vs CNN ───

def step_conformer_dicl_analysis(config, best_hp, best_alpha):
    """
    Best Conformer 모델에 대한:
      · 단일 모델 bootstrap CI (rho_all + rho_subset)
      · CNN baseline과의 paired bootstrap (Δρ + p-value)
    """
    print("\n" + "=" * 60)
    print(" Step W9: DICL Analysis (Bootstrap + Paired vs CNN)")
    print("=" * 60)

    import subprocess
    save_dir = config['train']['save_dir']
    output_dir = os.path.join(config['experiment']['result_directory'], save_dir, 'analysis')

    # Best 모델 = W2 α ablation에서 결정된 best_alpha 모델
    # (Multi-seed 제외 결정에 따라 단일 모델 사용)
    best_pt = os.path.join(
        config['experiment']['result_directory'], save_dir,
        'checkpoints', f'conf_alpha_{best_alpha}.pt'
    )
    cnn_pt = os.path.join('outputs', 'checkpoints', 'clean_baseline_cnn.pt')

    # 1) 단일 모델 bootstrap CI
    if os.path.exists(best_pt):
        print(f"\n  [Single model bootstrap CI] {best_pt}")
        subprocess.run([
            'python', '-m', 'analysis.dicl_correlation',
            '--config', 'configs/base_config.yaml',
            '--model_path', best_pt,
            '--output_dir', output_dir,
        ])
    else:
        print(f"  ⚠️ Best 모델 체크포인트 없음: {best_pt}")
        return

    # 2) Paired bootstrap (vs CNN)
    if os.path.exists(cnn_pt):
        print(f"\n  [Paired bootstrap] {best_pt} vs {cnn_pt}")
        subprocess.run([
            'python', '-m', 'analysis.dicl_correlation',
            '--config', 'configs/base_config.yaml',
            '--model_path', best_pt,
            '--compare_model', cnn_pt,
            '--output_dir', output_dir,
            '--n_boot', '1000',
        ])
    else:
        print(f"  ⚠️ CNN 체크포인트 없음: {cnn_pt} (paired 분석 skip)")


# ─── Weekend Full Pipeline (한 번 실행 → 전체 자동) ───

def run_weekend_full(config):
    """
    주말 동안 전체 자동 파이프라인.
    한 번 실행하면 W1 → W9까지 순차 자동 실행.
    각 step 실패해도 다음 step 진행 (try/except는 run_step 내부).
    """
    pipeline_start = time.time()
    print("\n" + "*" * 60)
    print(" 🚀 Weekend Full Pipeline (Conformer + Gate)")
    print("    한 번 실행하면 주말 동안 자동 진행")
    print("*" * 60)

    # 0. LR baseline (CPU, 가벼움 — 가장 먼저)
    try:
        step_lr_baseline(config)
    except Exception as e:
        print(f"  ⚠️ LR baseline 실패: {e}")

    # 1. Grid search (lr × layers) → best_hp
    best_hp = step_conformer_grid(config)
    print(f"\n  ★ best_hp: {best_hp}")

    # 2. α ablation → best_alpha
    best_alpha = step_conformer_alpha(config, best_hp)
    print(f"\n  ★ best_alpha: {best_alpha}")

    # 3~7: 모든 ablation
    step_conformer_channel(config, best_hp, best_alpha)
    step_conformer_conv_kernels(config, best_hp, best_alpha)
    step_conformer_fusion(config, best_hp, best_alpha)
    step_conformer_ensemble(config, best_hp, best_alpha)
    step_conformer_contrastive(config, best_hp, best_alpha)

    # 8: DICL 분석 + paired bootstrap (W2 best_alpha 모델 사용)
    # · Multi-seed는 시간 budget상 제외 — 별도 step('conf_multi_seed')로 수동 실행 가능
    step_conformer_dicl_analysis(config, best_hp, best_alpha)

    elapsed = time.time() - pipeline_start
    print(f"\n{'*' * 60}")
    print(f" 🎉 Weekend pipeline 완료 ({elapsed/3600:.1f}h)")
    print(f"{'*' * 60}")
    print(f"\n  확정된 best 설정:")
    print(f"    learning_rate:    {best_hp['learning_rate']}")
    print(f"    n_layers:         {best_hp['n_layers']}")
    print(f"    d_model:          {best_hp['d_model']}")
    print(f"    alpha:            {best_alpha}")
    print(f"\n  결과:")
    print(f"    outputs/{config['train']['save_dir']}/experiment_results.csv")
    print(f"    outputs/{config['train']['save_dir']}/analysis/rho_bootstrap_summary.csv")
    print(f"    outputs/{config['train']['save_dir']}/analysis/rho_paired_comparison.csv")


# ─── Paper Full Pipeline (구 run_paper_experiments.py 흡수) ───

def step_paper_full(config, config_path):
    """
    논문 전체 재현 파이프라인 (구 run_paper_experiments.py 통합).

    weekend_full(α 탐색용 grid)과 달리, 이미 확정된 best config
    (lr=3e-5, n_layers=4, α=0.3 — Position B)에서 논문 표/그림을 재현한다.

    실행 순서 (각 step → experiment_results.csv 누적):
      1. conf_conv_kernels  → Appendix kernel ablation (multi-seed에 없는 유일 ablation)
      2. multi_seed         → Table 1 Ours/CNN/+Contrast (3-seed) + Table 8
      3. multi_seed_extras  → Table 1 baselines(FT/BiLSTM/Transformer/Vanilla)
                              + Table 2 ablations(α 0.0/0.5/0.7, Gate) (3-seed)
      4. conf_dicl_analysis → DICL ρ + bootstrap CI + paired vs CNN
      5. run_paper_artifacts.py → t-SNE / attention / failure-mode 그림·표

    구 Phase A(baselines)·D(fusion)·E(ensemble)·B(α)는 multi_seed +
    multi_seed_extras가 3-seed로 포함하므로 별도 단계 제거.
    부분 재실행이 필요하면 개별 --step 사용.

    예상 ~80-130h (~3-5일).
    """
    import subprocess

    best_hp = {
        'learning_rate': float(config['train']['learning_rate']),
        'n_layers': config['model']['n_layers'],
        'd_model': config['model']['embedding_dim'],
    }
    best_alpha = float(config['train'].get('alpha', 0.3))

    pipeline_start = time.time()
    print("\n" + "*" * 60)
    print(" 🚀 Paper Full Pipeline (논문 재현)")
    print("*" * 60)

    # 1. Kernel ablation (Appendix)
    step_conformer_conv_kernels(config, best_hp, best_alpha)
    # 2. Multi-seed 메인 (Table 1 / Table 8)
    step_multi_seed(config)
    # 3. Multi-seed baselines + ablations (Table 1 baselines / Table 2)
    step_multi_seed_extras(config)
    # 4. DICL ρ + bootstrap
    step_conformer_dicl_analysis(config, best_hp, best_alpha)
    # 5. Figures / tables (별도 스크립트)
    try:
        subprocess.run(['python', 'run_paper_artifacts.py', '--config', config_path])
    except Exception as e:
        print(f"  ⚠️ paper_artifacts 실패: {e}")

    elapsed = time.time() - pipeline_start
    print(f"\n{'*' * 60}")
    print(f" 🎉 Paper full pipeline 완료 ({elapsed/3600:.1f}h)")
    print(f"{'*' * 60}")


# ════════════════════════════════════════════════════════════════════
#  메인
# ════════════════════════════════════════════════════════════════════

STEP_NAMES = [
    'lr_baseline', 'clean_rerun', 'syn_gated_combo',
    'fusion_linguistic', 'fusion_linguistic_typology',
    'mcc', 'mcc_tune', 'mcc_combo', 'mcc_convstem', 'mcc_arch_overnight',
    'multi_seed',   # ★ paper main result 신뢰구간 (Ours + CNN + Ours+Contrast)
    'multi_seed_extras',   # ★ Table 1 baselines (FT/BiLSTM/Transformer/Vanilla MCC) + Table 2 ablations (α 0.0/0.5/0.7, Gate)
    # ★ canonical Conformer 파이프라인
    'weekend_full',   # α 탐색 포함 전체 자동 (grid→ablation→DICL)
    'paper_full',     # 논문 재현 (확정 config, 구 run_paper_experiments.py)
    'conf_grid', 'conf_alpha', 'conf_channel',
    'conf_conv_kernels', 'conf_fusion', 'conf_ensemble',
    'conf_contrastive', 'conf_multi_seed', 'conf_dicl_analysis',
]


def main():
    parser = argparse.ArgumentParser(description="Run All Experiments (Auto-chained)")
    parser.add_argument("--config", type=str, default=os.path.join("configs", "base_config.yaml"))
    parser.add_argument("--step", type=str, default=None, choices=STEP_NAMES,
                        help="특정 단계만 실행")
    args = parser.parse_args()

    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    set_seed(config['experiment']['seed'])
    results_csv = os.path.join(
        config['experiment']['result_directory'],
        config['train']['save_dir'],
        'experiment_results.csv',
    )
    total_start = time.time()

    # ── Weekend full pipeline (한 번 실행 → 주말 자동, α 탐색 포함) ──
    if args.step == 'weekend_full':
        run_weekend_full(config)
        print(f"\n  완료 ({time.time() - total_start:.1f}s)")
        return

    # ── Paper full pipeline (논문 재현, 확정 config) ──
    if args.step == 'paper_full':
        step_paper_full(config, args.config)
        print(f"\n  완료 ({time.time() - total_start:.1f}s)")
        return

    # ── 기본 실행(--step 미지정): canonical 파이프라인 안내 ──
    # 옛 TE 기반 자동 체인은 제거됨(2026-05 리팩토링, conf_* 로 대체).
    if args.step is None:
        print("=" * 60)
        print(" --step 미지정. canonical 파이프라인을 선택하세요:")
        print("   --step paper_full          # 논문 재현 (확정 config, kernel→multi-seed→DICL→figures)")
        print("   --step weekend_full        # α 탐색 포함 전체 자동 (grid→ablation→DICL)")
        print("   --step multi_seed          # 3-seed 신뢰구간 (Ours/CNN/+Contrast)")
        print("   --step multi_seed_extras   # Table 1 baselines + Table 2 ablations")
        print(f"\n   사용 가능한 개별 step:\n     {', '.join(STEP_NAMES)}")
        print("=" * 60)
        return

    # ── 개별 Step 실행 ──
    else:
        print(f"  [개별 실행] Step: {args.step}\n")

        if args.step == 'lr_baseline':
            step_lr_baseline(config)
        elif args.step == 'clean_rerun':
            step_clean_rerun(config)
        elif args.step == 'syn_gated_combo':
            step_syn_gated_combo(config)
        elif args.step == 'fusion_linguistic' or args.step == 'fusion_linguistic_typology':
            step_fusion_linguistic(config)
        elif args.step == 'mcc':
            step_mcc(config)
        elif args.step == 'mcc_tune':
            step_mcc_tune(config)
        elif args.step == 'mcc_combo':
            step_mcc_combo(config)
        elif args.step == 'mcc_convstem':
            step_mcc_convstem(config)
        elif args.step == 'mcc_arch_overnight':
            step_mcc_arch_overnight(config)
        elif args.step == 'multi_seed':
            step_multi_seed(config)
        elif args.step == 'multi_seed_extras':
            step_multi_seed_extras(config)
        # ─── Conformer+Gate 개별 step (디버깅용) ───
        elif args.step == 'conf_grid':
            step_conformer_grid(config)
        elif args.step == 'conf_alpha':
            best_hp = {
                'learning_rate': float(config['train']['learning_rate']),
                'n_layers': config['model']['n_layers'],
                'd_model': config['model']['embedding_dim'],
            }
            step_conformer_alpha(config, best_hp)
        elif args.step == 'conf_channel':
            best_hp = {
                'learning_rate': float(config['train']['learning_rate']),
                'n_layers': config['model']['n_layers'],
                'd_model': config['model']['embedding_dim'],
            }
            step_conformer_channel(config, best_hp, float(config['train'].get('alpha', 0.5)))
        elif args.step == 'conf_conv_kernels':
            best_hp = {
                'learning_rate': float(config['train']['learning_rate']),
                'n_layers': config['model']['n_layers'],
                'd_model': config['model']['embedding_dim'],
            }
            step_conformer_conv_kernels(config, best_hp, float(config['train'].get('alpha', 0.5)))
        elif args.step == 'conf_fusion':
            best_hp = {
                'learning_rate': float(config['train']['learning_rate']),
                'n_layers': config['model']['n_layers'],
                'd_model': config['model']['embedding_dim'],
            }
            step_conformer_fusion(config, best_hp, float(config['train'].get('alpha', 0.5)))
        elif args.step == 'conf_ensemble':
            best_hp = {
                'learning_rate': float(config['train']['learning_rate']),
                'n_layers': config['model']['n_layers'],
                'd_model': config['model']['embedding_dim'],
            }
            step_conformer_ensemble(config, best_hp, float(config['train'].get('alpha', 0.5)))
        elif args.step == 'conf_contrastive':
            best_hp = {
                'learning_rate': float(config['train']['learning_rate']),
                'n_layers': config['model']['n_layers'],
                'd_model': config['model']['embedding_dim'],
            }
            step_conformer_contrastive(config, best_hp, float(config['train'].get('alpha', 0.5)))
        elif args.step == 'conf_multi_seed':
            best_hp = {
                'learning_rate': float(config['train']['learning_rate']),
                'n_layers': config['model']['n_layers'],
                'd_model': config['model']['embedding_dim'],
            }
            step_conformer_multi_seed(config, best_hp, float(config['train'].get('alpha', 0.5)))
        elif args.step == 'conf_dicl_analysis':
            best_hp = {
                'learning_rate': float(config['train']['learning_rate']),
                'n_layers': config['model']['n_layers'],
                'd_model': config['model']['embedding_dim'],
            }
            step_conformer_dicl_analysis(config, best_hp, float(config['train'].get('alpha', 0.5)))

        print(f"\n  완료 ({time.time() - total_start:.1f}s)")


if __name__ == '__main__':
    main()
