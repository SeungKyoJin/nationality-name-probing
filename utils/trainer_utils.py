import numpy as np
import torch


class EarlyStopping:
    def __init__(self, patience=10, verbose=False, delta=0, path='best_model.pth'):
        """
        patience: 성능 향상이 없어도 참을 에포크 횟수
        verbose: 로그 출력 여부
        delta: 이 수치보다 크게 개선되어야 성능 향상으로 인정
        path: 베스트 모델 가중치를 저장할 경로
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta
        self.path = path

    def __call__(self, val_loss, model):
        # 점수는 낮을수록 좋으므로 loss에 -를 붙여 계산
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            # 성능 향상이 없으면 카운트 증가
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            # 성능 향상이 있으면 카운트 초기화 및 모델 저장
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss
