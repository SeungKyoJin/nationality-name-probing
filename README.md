# Linguistic Demography in Name Embeddings

Code for the paper **"Linguistic Demography in Name Embeddings: Geometry and Classification Difficulty"** (Findings of EACL 2027).

We train seven character-level classifiers that map a romanized personal name to one of 168 National Olympic Committees (NOCs) and one of 39 language groups, and probe the learned country-centroid geometry against the eleven indices of the DICL database.

## Intended use

This repository is released **for research purposes only**. Name-based nationality classification has documented misuse potential in employment, immigration, and surveillance settings. We do not endorse operational deployment for screening, profiling, or border-control decisions. No trained checkpoints and no individual-name predictions are distributed.

## Setup

```bash
pip install -r requirements.txt
```

PyTorch 2.0+ with CUDA is recommended. The reference model (~24M parameters) trains in about 4 hours on a single NVIDIA RTX 3080.

## Data

The corpus itself is not distributed, but it can be rebuilt from public sources:

1. **Olympic athlete records** — Kaggle, "120 years of Olympic history: athletes and results": https://www.kaggle.com/datasets/heesoo37/120-years-of-olympic-history-athletes-and-results
2. **Wikipedia personal names** — via the `name2nat` resource: https://github.com/Kyubyong/name2nat
3. **DICL v2** (linguistic indices, CC-BY 4.0) — Gurevich, Herman, Toubal, and Yotov (2025), *A Dataset on Linguistic Connectivity Across and Within Countries*, Scientific Data 12:542. https://doi.org/10.1038/s41597-025-04692-8

Merge the two name sources into `data/raw/concat_names_mapped.csv` with columns `Name, NOC, dataset, LanguageGroup` (cap 3,000 and minimum 100 names per NOC), and place the DICL file at `data/linguistics/DICL_v2.csv`. Paths are configurable in `configs/base_config.yaml`.

## Usage

```bash
# single training run
python train.py --config configs/base_config.yaml --mode train

# full paper pipeline (training, ablations, DICL analysis, figures)
python run_all_experiments.py --config configs/base_config.yaml --step paper_full

# TF-IDF logistic-regression baseline
python run_baseline_lr.py

# embedding-DICL correlation and isolation-F1 analyses
python -m analysis.dicl_correlation
python -m analysis.isolation_f1_cross_arch --config configs/base_config.yaml
```

## Citation

```bibtex
@inproceedings{jin2027linguistic,
  title     = {Linguistic Demography in Name Embeddings: Geometry and Classification Difficulty},
  author    = {Jin, Seungkyo and Shin, Jaemin and Han, Sangjun},
  booktitle = {Findings of the Association for Computational Linguistics: EACL 2027},
  year      = {2027}
}
```

## License

MIT (see `LICENSE`). The DICL database is used under CC-BY 4.0; Olympic and Wikipedia name sources are governed by their own licenses.
