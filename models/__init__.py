import json
import os

from .transformer import Encoder, MultiChannelEncoder, MultiChannelConformer
from .baselines import MultiChannelCNN, MultiChannelBiLSTM, MultiChannelFastText

MODEL_MAP = {
    'transformer_encoder': Encoder,
    'multichannel_encoder': MultiChannelEncoder,
    'multichannel_cnn': MultiChannelCNN,
    'multichannel_bilstm': MultiChannelBiLSTM,
    'multichannel_fasttext': MultiChannelFastText,
    'multichannel_conformer': MultiChannelConformer,   # ★ Architecture novelty
}


def _load_typology_and_idx_map(config):
    """
    Linguistic fusion에서 사용할 (typology_init, lang_idx_map) 로드.

    · typology_init: data/linguistics/typology_prior.json 또는 config 지정 경로
    · lang_idx_map:  data.get_lang_group_idx(config)

    fusion_type != 'linguistic'이거나 JSON 없으면 (None, None) 반환 → uniform fallback.
    """
    fusion_type = config.get('model', {}).get('fusion_type', 'linear')
    if fusion_type != 'linguistic':
        return None, None

    typology_path = config.get('model', {}).get(
        'typology_path',
        'data/linguistics/typology_prior.json',
    )
    if not os.path.exists(typology_path):
        print(f"[get_model] typology JSON 없음 ({typology_path}) → uniform 초기화")
        return None, None

    try:
        with open(typology_path, 'r', encoding='utf-8') as f:
            typology = json.load(f)
        from data import get_lang_group_idx
        lang_idx = get_lang_group_idx(config)
        print(f"[get_model] typology prior 로드: {typology_path} ({len(typology)} groups)")
        return typology, lang_idx
    except Exception as e:
        print(f"[get_model] typology 로드 실패: {e} → uniform fallback")
        return None, None


def get_model(config, vocab_size, n_classes, n_lang_groups):
    model_type = config['model']['type']

    if model_type not in MODEL_MAP:
        raise ValueError(
            f"지원하지 않는 모델 아키텍처입니다: {model_type}\n"
            f"사용 가능: {list(MODEL_MAP.keys())}"
        )

    model_class = MODEL_MAP[model_type]

    base_kwargs = dict(
        n_classes=n_classes,
        n_lang_groups=n_lang_groups,
        n_head=config['model']['n_head'],
        d_model=config['model']['embedding_dim'],
        d_hid=config['model']['hidden_dim'],
        n_layers=config['model']['n_layers'],
        vocab_size=vocab_size,
        dropout=config['model'].get('dropout', 0.5),
    )

    # MultiChannelEncoder 전용 옵션 (fusion/pooling/typology)
    if model_type == 'multichannel_encoder':
        base_kwargs['fusion_type'] = config['model'].get('fusion_type', 'linear')
        base_kwargs['pool_type'] = config['model'].get('pool_type', 'attention')
        # linguistic fusion일 때만 typology 로드 (다른 fusion은 None 처리)
        typology, lang_idx = _load_typology_and_idx_map(config)
        if typology is not None:
            base_kwargs['typology_init'] = typology
            base_kwargs['lang_idx_map'] = lang_idx

    # MultiChannelConformer 전용 옵션 (channel-wise conv kernel + pool + cross-attn + stem)
    if model_type == 'multichannel_conformer':
        base_kwargs['pool_type'] = config['model'].get('pool_type', 'attention')
        # kernel sizes (uni, bi, tri) — 각 n-gram scale에 맞춤
        kernels = config['model'].get('conv_kernels', [2, 3, 5])
        base_kwargs['conv_kernels'] = tuple(kernels)
        # Cross-attn 주기: False=기존(마지막 1회) | True=매 layer (channel-fragmentation 완화)
        base_kwargs['cross_attn_per_layer'] = bool(
            config['model'].get('cross_attn_per_layer', False)
        )
        # Conv stem: CNN inductive bias 주입 (Embedding 직후, attention 전)
        base_kwargs['conv_stem'] = bool(config['model'].get('conv_stem', False))
        base_kwargs['conv_stem_kernel'] = int(config['model'].get('conv_stem_kernel', 5))
        base_kwargs['conv_stem_layers'] = int(config['model'].get('conv_stem_layers', 2))
        # Stem type: 'standard' (단일 kernel) | 'multiscale' (Inception-style)
        base_kwargs['conv_stem_type'] = config['model'].get('conv_stem_type', 'standard')
        ms_kernels = config['model'].get('conv_stem_kernels', [3, 5, 7])
        base_kwargs['conv_stem_kernels'] = tuple(ms_kernels)
        # Channel ensemble head: cross-attn 우회, 채널별 독립 dual head + fusion
        base_kwargs['channel_ensemble_head'] = bool(
            config['model'].get('channel_ensemble_head', False)
        )
        # Channel fusion type: 'uniform' (1/3 평균) | 'gate' (MoE-style 입력 의존 가중치)
        base_kwargs['channel_fusion_type'] = config['model'].get(
            'channel_fusion_type', 'uniform'
        )

    return model_class(**base_kwargs)
