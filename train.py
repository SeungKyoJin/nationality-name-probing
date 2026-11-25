import os
import time
import argparse
import warnings
from datetime import datetime

warnings.filterwarnings('ignore', category=FutureWarning)

import yaml

from data import (
    prepare_data, get_num_labels, get_num_lang_labels, get_vocab_size,
)
from models import get_model
from utils import merge_config, save_result
from trainer import Trainer
from inference import run_inference


os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def run_training(config):
    """config 하나에 대해 학습 1회 수행. run_all_experiments.py의 step들이 반복 호출."""
    # save_dir에 날짜 폴더 자동 추가 (예: checkpoints/260422/)
    date_dir = datetime.now().strftime("%y%m%d")
    config['train']['save_dir'] = os.path.join(config['train']['save_dir'], date_dir)

    num_labels = get_num_labels(config)
    num_lang_labels = get_num_lang_labels(config)
    vocab_size = get_vocab_size(config)

    train_loader, valid_loader, test_loader, nat_weights, lang_weights = prepare_data(config)

    model = get_model(
        config, vocab_size,
        n_classes=config['model']['num_classes'],
        n_lang_groups=config['model']['num_lang_groups'],
    )

    trainer = Trainer(model, config, num_labels, num_lang_labels,
                      nat_weights=nat_weights, lang_weights=lang_weights)
    metrics = trainer.fit(train_loader, valid_loader, test_loader)

    # Auto-log CNL Spearman ρ for multichannel_encoder (Phase 4 정량 evidence)
    # 학습 완료 후 saved checkpoint로 CNL 상관 빠르게 계산 → metrics에 merge
    saved_model_path = os.path.join(
        config['experiment']['result_directory'],
        config['train']['save_dir'],
        config['train']['save_name'],
    )
    try:
        from analysis.dicl_correlation import compute_cnl_rho_quick
        cnl_metrics = compute_cnl_rho_quick(config, saved_model_path)
        if cnl_metrics:
            metrics.update(cnl_metrics)
            print(f"[CNL auto-log] ρ_all={cnl_metrics['cnl_rho_all']:.4f}  "
                  f"ρ_subset={cnl_metrics['cnl_rho_subset']:.4f}  "
                  f"n={cnl_metrics['cnl_n_pairs']}")
    except Exception as e:
        print(f"[WARN] CNL auto-log 실패 (학습은 정상): {e}")

    save_result(config, metrics)
    return metrics


def main():
    start = time.time()

    parser = argparse.ArgumentParser(description="Nationality Classification")
    parser.add_argument("--config", type=str, default=os.path.join("configs", "base_config.yaml"))
    parser.add_argument("--mode", type=str, default=None, choices=[None, 'train', 'inference'])
    parser.add_argument("--model_type", type=str, default=None)
    parser.add_argument("--ngram", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight_decay", type=float, default=None)
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--embedding_dim", type=int, default=None)
    parser.add_argument("--hidden_dim", type=int, default=None)
    parser.add_argument("--num_layers", type=int, default=None)
    parser.add_argument("--max_length", type=int, default=None)
    parser.add_argument("--save_dir", type=str, default=None)
    parser.add_argument("--save_name", type=str, default=None)
    args = parser.parse_args()

    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    if args.mode is not None:
        config['experiment']['mode'] = args.mode
    config = merge_config(config, args)

    mode = config['experiment']['mode']
    if mode == 'train':
        run_training(config)
    elif mode == 'inference':
        run_inference(config)
    else:
        raise ValueError(f"알 수 없는 mode: {mode}. 'train' 또는 'inference' 중 선택하세요.")

    print(f"Total time: {time.time() - start:.2f}s")


if __name__ == '__main__':
    main()
