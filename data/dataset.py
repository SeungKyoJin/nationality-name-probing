from torch.utils.data import Dataset
from utils import util

import torch


_NGRAM_CONVERTERS = {
    'uni': util.text_to_unigram_indices,
    'bi': util.text_to_bigram_indices,
    'tri': util.text_to_trigram_indices,
    'mixed': util.text_to_mixed_indices,
}


class TextDataset(Dataset):
    """
    Single-channel (uni/bi/tri/mixed) n-gram dataset.
    반환: (indices, nat_label, lang_label)
    """

    def __init__(self, texts, char_to_idx, max_length, country_idx, lang_group_idx,
                 n_gram=None, labels=None, lang_labels=None):
        if n_gram not in _NGRAM_CONVERTERS:
            raise ValueError(f"지원하지 않는 n_gram: {n_gram}")

        self.texts = texts
        self.labels = labels
        self.lang_labels = lang_labels
        self.char_to_idx = char_to_idx
        self.max_length = max_length
        self.country_idx = country_idx
        self.lang_group_idx = lang_group_idx
        self.n_gram = n_gram
        self._convert = _NGRAM_CONVERTERS[n_gram]

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts.iloc[idx]
        nat_label = self.labels.iloc[idx]
        lang_label = self.lang_labels.iloc[idx]

        indices = self._convert(text, self.char_to_idx, self.max_length)
        return (
            torch.LongTensor(indices),
            torch.tensor(self.country_idx[nat_label]),
            torch.tensor(self.lang_group_idx[lang_label]),
        )


class MultiChannelDataset(Dataset):
    """
    Multi-channel (uni + bi + tri 동시) 데이터셋. 주력 모델 용도.
    반환: (uni, bi, tri, nat_label, lang_label)
    """

    def __init__(self, texts, labels, lang_labels,
                 uni_ngram_dict, bi_ngram_dict, tri_ngram_dict,
                 max_length, country_idx, lang_group_idx):
        self.texts = texts
        self.labels = labels
        self.lang_labels = lang_labels
        self.uni_ngram_dict = uni_ngram_dict
        self.bi_ngram_dict = bi_ngram_dict
        self.tri_ngram_dict = tri_ngram_dict
        self.max_length = max_length
        self.country_idx = country_idx
        self.lang_group_idx = lang_group_idx

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts.iloc[idx]
        nat_label = self.labels.iloc[idx]
        lang_label = self.lang_labels.iloc[idx]

        uni = util.text_to_unigram_indices(text, self.uni_ngram_dict, self.max_length)
        bi = util.text_to_bigram_indices(text, self.bi_ngram_dict, self.max_length)
        tri = util.text_to_trigram_indices(text, self.tri_ngram_dict, self.max_length)

        return (
            torch.LongTensor(uni),
            torch.LongTensor(bi),
            torch.LongTensor(tri),
            torch.tensor(self.country_idx[nat_label]),
            torch.tensor(self.lang_group_idx[lang_label]),
        )


class InferenceDataset(Dataset):
    """추론용 (라벨 없음)."""

    def __init__(self, texts, char_to_idx, max_length):
        self.texts = texts
        self.char_to_idx = char_to_idx
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts.iloc[idx]
        indices = util.text_to_trigram_indices(text, self.char_to_idx, self.max_length)
        return torch.LongTensor(indices)
