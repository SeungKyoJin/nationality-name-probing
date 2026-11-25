import inspect
import math
import os

import torch
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import LambdaLR
import pandas as pd
from torch import nn
from sklearn.metrics import classification_report, f1_score

from utils.trainer_utils import EarlyStopping
from utils.losses import SupConLoss
from data import get_country_list, get_lang_group_list


class Trainer:
    """
    Hierarchical classification trainer with optional contrastive auxiliary loss.

    Loss = α · L_lang + (1 - α) · L_nat  +  γ · L_contrastive   (γ=0이면 기존과 동일)
      - α=0: nationality only
      - α=1: language group only
      - 0<α<1: dual head
      - γ=0: contrastive off (기존 동작)
      - γ>0: SupCon 추가 (variant: nat | lang | hier)
    """

    def __init__(self, model, config, num_labels, num_lang_labels,
                 nat_weights=None, lang_weights=None):
        self.config = config
        self.device = config['experiment']['device']
        if self.device == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA 사용 불가. GPU 환경을 확인하세요:\n"
                "  1. nvidia-smi 로 GPU 확인\n"
                "  2. python -c \"import torch; print(torch.cuda.is_available())\" 로 PyTorch CUDA 확인\n"
                "  3. conda install pytorch pytorch-cuda=11.8 -c pytorch -c nvidia 로 재설치"
            )
        self.m = model.to(self.device)
        self.num_labels = num_labels                # [0, ..., n_classes-1]
        self.num_lang_labels = num_lang_labels      # [0, ..., n_lang_groups-1]

        self.batch_size = int(config['train']['batch_size'])
        self.lr = float(config['train']['learning_rate'])
        self.weight_decay = float(config['train']['weight_decay'])
        self.epochs = int(config['train']['epochs'])
        self.alpha = float(config['train'].get('alpha', 0.5))

        # Class-weighted CrossEntropyLoss: 소수 클래스에 더 높은 가중치
        if nat_weights is not None:
            nat_weights = nat_weights.to(self.device)
            print(f"[Trainer] Nationality class weights 적용 ({len(nat_weights)} classes)")
        if lang_weights is not None:
            lang_weights = lang_weights.to(self.device)
            print(f"[Trainer] LanguageGroup class weights 적용 ({len(lang_weights)} groups)")

        # Label smoothing — 168-class CE에서 calibration + macro F1 개선 (Szegedy 2016)
        self.label_smoothing = float(config['train'].get('label_smoothing', 0.0))
        if self.label_smoothing > 0:
            print(f"[Trainer] Label smoothing 활성화 (ε={self.label_smoothing})")

        self.criterion_nat = nn.CrossEntropyLoss(
            weight=nat_weights, label_smoothing=self.label_smoothing
        )
        self.criterion_lang = nn.CrossEntropyLoss(
            weight=lang_weights, label_smoothing=self.label_smoothing
        )

        # Contrastive auxiliary loss (γ=0이면 사용 안 함)
        self.contrastive_weight = float(config['train'].get('contrastive_weight', 0.0))
        self.contrastive_variant = config['train'].get('contrastive_variant', 'hier')
        if self.contrastive_weight > 0:
            self.criterion_contrast = SupConLoss(
                temperature=float(config['train'].get('contrastive_temperature', 0.07)),
                variant=self.contrastive_variant,
                lang_weight=float(config['train'].get('contrastive_lang_weight', 0.3)),
            )
            print(f"[Trainer] Contrastive loss 활성화 "
                  f"(γ={self.contrastive_weight}, variant={self.contrastive_variant})")
        else:
            self.criterion_contrast = None

        # Linguistic-aware fusion 사용 시 auxiliary lang loss
        # MultiChannelEncoder가 fusion_type='linguistic'이면 forward 후
        # self.m._last_aux_lang_logits 에 [B, n_lang] logits이 stash됨.
        self.aux_lang_weight = float(config['train'].get('aux_lang_weight', 0.1))
        is_linguistic_fusion = (
            config.get('model', {}).get('fusion_type') == 'linguistic'
        )
        if is_linguistic_fusion and self.aux_lang_weight > 0:
            self.criterion_aux_lang = nn.CrossEntropyLoss(weight=lang_weights)
            print(f"[Trainer] Linguistic-aware fusion → aux lang loss "
                  f"(β={self.aux_lang_weight})")
        else:
            self.criterion_aux_lang = None

        # Model이 return_features=True 지원하는지 확인.
        # 클래스 이름 == 비교는 MultiChannelConformer 등 신규 모델을 놓치므로,
        # forward signature를 inspect해서 'return_features' 파라미터 유무로 판정.
        # Why: 과거 type(...).__name__ == 'MultiChannelEncoder' 로 하드체크하던 코드는
        # Conformer에서 contrastive를 silent하게 무력화시키는 버그였음 (W7 ablation 무효).
        try:
            forward_params = inspect.signature(self.m.forward).parameters
            self._model_returns_features = 'return_features' in forward_params
        except (ValueError, TypeError):
            self._model_returns_features = False
        if self.contrastive_weight > 0 and not self._model_returns_features:
            print(f"[WARN] {type(self.m).__name__}.forward는 return_features 미지원 → "
                  f"contrastive 비활성화")
            self.contrastive_weight = 0.0
            self.criterion_contrast = None

        self.optimizer = optim.AdamW(
            self.m.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )

        # LR schedule — 'fixed' (기존) | 'cosine_warmup' (Conformer/BERT 표준)
        # cosine_warmup은 fit() 시작 시 total_steps 알게 되어 build됨
        self.lr_schedule = config['train'].get('lr_schedule', 'fixed')
        self.warmup_ratio = float(config['train'].get('warmup_ratio', 0.05))
        self.scheduler = None  # fit() 시작 시 build
        if self.lr_schedule == 'cosine_warmup':
            print(f"[Trainer] LR schedule = cosine_warmup "
                  f"(peak={self.lr}, warmup_ratio={self.warmup_ratio})")

        save_path = os.path.join(
            config['experiment']['result_directory'],
            config['train']['save_dir'],
            config['train']['save_name'],
        )
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        self.early_stopping = EarlyStopping(
            patience=config['train'].get('patience', 10),
            verbose=True,
            path=save_path,
        )

        # Mixed Precision (FP16)
        self.use_amp = (self.device == 'cuda' and torch.cuda.is_available())
        self.scaler = GradScaler(enabled=self.use_amp)
        if self.use_amp:
            print("[Trainer] Mixed Precision (FP16) 활성화")

    # ------------------------------------------------------------------
    # 학습
    # ------------------------------------------------------------------
    def _build_cosine_warmup_scheduler(self, total_steps, warmup_steps):
        """LambdaLR로 warmup(linear) → cosine decay 구성."""
        def lr_lambda(step):
            if step < warmup_steps:
                return step / max(1, warmup_steps)
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            return 0.5 * (1.0 + math.cos(math.pi * progress))
        return LambdaLR(self.optimizer, lr_lambda)

    def fit(self, train_loader, valid_loader, test_loader):
        # cosine_warmup scheduler — total_steps는 train_loader 길이로 산정
        if self.lr_schedule == 'cosine_warmup' and self.scheduler is None:
            total_steps = len(train_loader) * self.epochs
            warmup_steps = int(total_steps * self.warmup_ratio)
            self.scheduler = self._build_cosine_warmup_scheduler(total_steps, warmup_steps)
            print(f"[Trainer] Scheduler built: total_steps={total_steps}, "
                  f"warmup_steps={warmup_steps}")

        for epoch in range(self.epochs):
            self.m.train()
            if epoch % 10 == 0:
                print(f'[Epoch {epoch}]')

            train_loss = 0.0
            n_samples = 0

            for batch in train_loader:
                inputs, nat_target, lang_target = self._unpack_batch(batch)
                self.optimizer.zero_grad()

                with autocast(enabled=self.use_amp):
                    # Contrastive 사용 시 pooled feature도 함께 반환
                    if self.contrastive_weight > 0:
                        nat_logits, lang_logits, pooled = self.m(
                            *inputs, return_features=True
                        )
                    else:
                        nat_logits, lang_logits = self.m(*inputs)
                        pooled = None

                    loss_nat = self.criterion_nat(nat_logits, nat_target)
                    loss_lang = self.criterion_lang(lang_logits, lang_target)
                    loss = self.alpha * loss_lang + (1.0 - self.alpha) * loss_nat

                    # Contrastive auxiliary loss
                    if pooled is not None and self.criterion_contrast is not None:
                        loss_contrast = self.criterion_contrast(
                            pooled, nat_target, lang_target
                        )
                        loss = loss + self.contrastive_weight * loss_contrast

                    # Linguistic-aware fusion auxiliary lang loss
                    if self.criterion_aux_lang is not None:
                        aux_logits = getattr(self.m, '_last_aux_lang_logits', None)
                        if aux_logits is not None:
                            loss_aux = self.criterion_aux_lang(aux_logits, lang_target)
                            loss = loss + self.aux_lang_weight * loss_aux

                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.m.parameters(), max_norm=1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()

                # LR scheduler step (per batch — warmup/cosine은 step 단위)
                if self.scheduler is not None:
                    self.scheduler.step()

                bsz = nat_target.size(0)
                train_loss += loss.item() * bsz
                n_samples += bsz

            print(f"Train loss: {train_loss / n_samples:.4f}")

            valid_metrics = self.evaluate(valid_loader)
            print(
                f"Valid | loss={valid_metrics['loss']:.4f} | "
                f"Nat Top1={valid_metrics['nat_top1']:.2f}% "
                f"Top5={valid_metrics['nat_top5']:.2f}% "
                f"MacroF1={valid_metrics['nat_macro_f1']:.2f}% || "
                f"Lang Top1={valid_metrics['lang_top1']:.2f}% "
                f"MacroF1={valid_metrics['lang_macro_f1']:.2f}%"
            )

            self.early_stopping(valid_metrics['loss'], self.m)
            if self.early_stopping.early_stop:
                print("[EarlyStopping] 학습을 조기 종료합니다.")
                break

        # Best checkpoint 로드 후 테스트
        self.m.load_state_dict(torch.load(self.early_stopping.path))
        print("\n[Test 평가]")
        test_metrics = self.evaluate(test_loader)
        self._print_metrics(test_metrics, tag="Test")
        self._save_reports(test_metrics)

        return test_metrics

    # ------------------------------------------------------------------
    # 평가 (5개 지표)
    # ------------------------------------------------------------------
    def evaluate(self, loader):
        self.m.eval()

        total_loss = 0.0
        total = 0
        correct_nat = 0
        top5_correct = 0
        correct_lang = 0
        lang_top3_correct = 0
        lang_top5_correct = 0

        y_true_nat, y_pred_nat = [], []
        y_true_lang, y_pred_lang = [], []

        with torch.no_grad():
            for batch in loader:
                inputs, nat_target, lang_target = self._unpack_batch(batch)
                nat_logits, lang_logits = self.m(*inputs)

                loss_nat = self.criterion_nat(nat_logits, nat_target)
                loss_lang = self.criterion_lang(lang_logits, lang_target)
                loss = self.alpha * loss_lang + (1.0 - self.alpha) * loss_nat

                bsz = nat_target.size(0)
                total_loss += loss.item() * bsz
                total += bsz

                # Nationality: Top-1 / Top-5
                _, top5_idx = torch.topk(nat_logits, k=5, dim=1)
                nat_pred = top5_idx[:, 0]
                correct_nat += (nat_pred == nat_target).sum().item()
                top5_correct += (top5_idx == nat_target.view(-1, 1)).any(dim=1).sum().item()

                # Language group: Top-1 / Top-3 / Top-5
                lang_pred = lang_logits.argmax(dim=1)
                correct_lang += (lang_pred == lang_target).sum().item()

                _, lang_top3_idx = torch.topk(lang_logits, k=3, dim=1)
                lang_top3_correct += (lang_top3_idx == lang_target.view(-1, 1)).any(dim=1).sum().item()

                _, lang_top5_idx = torch.topk(lang_logits, k=5, dim=1)
                lang_top5_correct += (lang_top5_idx == lang_target.view(-1, 1)).any(dim=1).sum().item()

                y_true_nat.extend(nat_target.cpu().numpy())
                y_pred_nat.extend(nat_pred.cpu().numpy())
                y_true_lang.extend(lang_target.cpu().numpy())
                y_pred_lang.extend(lang_pred.cpu().numpy())

        # ---- 지표 집계 ----
        nat_top1 = correct_nat / total * 100
        nat_top5 = top5_correct / total * 100
        lang_top1 = correct_lang / total * 100
        lang_top3 = lang_top3_correct / total * 100
        lang_top5 = lang_top5_correct / total * 100

        nat_macro_f1 = f1_score(
            y_true_nat, y_pred_nat,
            labels=self.num_labels, average='macro', zero_division=0,
        ) * 100
        nat_weighted_f1 = f1_score(
            y_true_nat, y_pred_nat,
            labels=self.num_labels, average='weighted', zero_division=0,
        ) * 100
        lang_macro_f1 = f1_score(
            y_true_lang, y_pred_lang,
            labels=self.num_lang_labels, average='macro', zero_division=0,
        ) * 100
        lang_weighted_f1 = f1_score(
            y_true_lang, y_pred_lang,
            labels=self.num_lang_labels, average='weighted', zero_division=0,
        ) * 100

        # 저장용 classification report (per-class precision/recall/f1)
        country_list = get_country_list(self.config)
        lang_list = get_lang_group_list(self.config)
        nat_report = classification_report(
            y_true_nat, y_pred_nat,
            labels=self.num_labels, target_names=country_list,
            output_dict=True, zero_division=0,
        )
        lang_report = classification_report(
            y_true_lang, y_pred_lang,
            labels=self.num_lang_labels, target_names=lang_list,
            output_dict=True, zero_division=0,
        )

        return {
            'loss': total_loss / total,
            'nat_top1': nat_top1,
            'nat_top5': nat_top5,
            'nat_macro_f1': nat_macro_f1,
            'nat_weighted_f1': nat_weighted_f1,
            'lang_top1': lang_top1,
            'lang_top3': lang_top3,
            'lang_top5': lang_top5,
            'lang_macro_f1': lang_macro_f1,
            'lang_weighted_f1': lang_weighted_f1,
            'nat_report': nat_report,
            'lang_report': lang_report,
        }

    # ------------------------------------------------------------------
    # 헬퍼
    # ------------------------------------------------------------------
    def _unpack_batch(self, batch):
        """batch = (*inputs, nat_target, lang_target) 형태."""
        *inputs, nat_target, lang_target = batch
        inputs = [t.to(self.device) for t in inputs]
        nat_target = nat_target.to(self.device)
        lang_target = lang_target.to(self.device)
        return inputs, nat_target, lang_target

    def _print_metrics(self, metrics, tag="Test"):
        print(
            f"[{tag}] loss={metrics['loss']:.4f}\n"
            f"  Nationality: Top-1={metrics['nat_top1']:.2f}% | "
            f"Top-5={metrics['nat_top5']:.2f}% | "
            f"Macro-F1={metrics['nat_macro_f1']:.2f}% | "
            f"Weighted-F1={metrics['nat_weighted_f1']:.2f}%\n"
            f"  Language:    Top-1={metrics['lang_top1']:.2f}% | "
            f"Top-3={metrics['lang_top3']:.2f}% | "
            f"Top-5={metrics['lang_top5']:.2f}% | "
            f"Macro-F1={metrics['lang_macro_f1']:.2f}% | "
            f"Weighted-F1={metrics['lang_weighted_f1']:.2f}%"
        )

    def _save_reports(self, metrics):
        base = os.path.join(
            self.config['experiment']['result_directory'],
            self.config['train']['save_dir'],
        )
        save_name = self.config['train']['save_name']

        nat_df = pd.DataFrame(metrics['nat_report']).transpose()
        lang_df = pd.DataFrame(metrics['lang_report']).transpose()

        nat_path = os.path.join(base, f"class_report_nat_{save_name}.csv")
        lang_path = os.path.join(base, f"class_report_lang_{save_name}.csv")

        nat_df.to_csv(nat_path, index=True)
        lang_df.to_csv(lang_path, index=True)
        print(f"[Report] 저장: {nat_path}, {lang_path}")
