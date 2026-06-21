# HTmiR — Handwritten Text Recognition for Medieval French (13th c.)

*[Version française](README-FR.md)*

Fine-tuning of a Kraken HTR model for the automatic transcription of
13th-century French manuscripts (Gothic *textualis*), trained on the CATMuS
Medieval corpus. This repository covers the Computer Vision component
(segmentation and recognition); the NLP post-processing component is documented
separately.

**Model (HuggingFace):** https://huggingface.co/abdessamadtouzani/htmir-french-13c
**Live demo (Streamlit):** https://htmirproject.streamlit.app
**Research article:** `paper/article.tex` (IEEE format, French)

---

## Results

| Metric | Value | 95% CI (bootstrap, N=1000) |
|---|---|---|
| CER (character error rate, validation) | 4.47% | [3.7%, 5.3%] |
| WER (word error rate, validation) | 23.6% | [21.0%, 26.2%] |
| Character accuracy | 95.5% | — |
| Line segmentation IoU (HTRomance, independent set) | 78.5% | — |

Target was CER < 8%. Full methodology and discussion in `paper/`.

---

## 1. Motivation

The project originally targeted the transcription of Leonardo da Vinci's
mirror-script notebooks. The blocking constraint is that **no aligned ground
truth** (image/transcription pairs) is publicly available for these manuscripts,
while supervised HTR fine-tuning requires exactly such pairs. This is compounded
by the recent publication of additional da Vinci material (notebook folios and
annotated drawings released only days before this work began), which is not yet
accompanied by machine-usable transcriptions.

Two consequences follow. First, ground truth for da Vinci could realistically be
**produced with [eScriptorium](https://gitlab.com/scripta/escriptorium)** — the
open-source transcription/annotation platform built on Kraken — through an
assisted annotation campaign; this is a project in itself and out of the present
scope. Second, rather than annotating da Vinci pages by hand within the available
time, we first build and validate a complete, reproducible HTR pipeline on a
richly annotated medieval corpus — **13th-century French** — whose Gothic script
shares with da Vinci's hand the density, abbreviations and ligatures that make
ancient HTR difficult. The pipeline is then directly reusable for a future
da Vinci ground-truth and fine-tuning campaign.

## 2. Data

- **Source:** CATMuS Medieval (HuggingFace `CATMuS/medieval`), filtered on
  `language = French`, `century = 13`.
- **Volumetry:** ~21,600 lines — 19,238 train / 969 validation / 1,369 test.
  Alphabet of 252 symbols (medieval abbreviations preserved).
- **Extraction:** queried via DuckDB (predicate pushdown) to avoid downloading
  the full ~25 GB of parquet files; only the relevant lines are materialized.

### Storage: cloud (S3), no local copy

The training data is **not stored in this repository nor kept locally**. The
source of truth is **AWS S3** (`s3://htmir-data/datasets/catmus-french-13c`).
The data is **downloaded at training time**:

```
training start
   -> download_dataset(bucket="htmir-data", prefix="datasets/catmus-french-13c")   [src/htmir/data/s3_sync.py]
   -> if absent from S3: prepare from HuggingFace (CATMuS) then push to S3 (idempotent)
   -> ketos compile + ketos train
```

This logic runs in both entry points: `src/htmir/cli/run_local.py` (local GPU)
and `infrastructure/train_entrypoint.py` (SageMaker).

## 3. Pipeline

```
CATMuS Medieval (HuggingFace)  ->  S3 (source of truth)
        |
        v
prepare_catmus.py   ->  line images (.png) + transcriptions (.gt.txt)
        |
        v
train_kraken.py     ->  fine-tuning from the CATMuS Medieval base model (ketos)
        |
        v
evaluate.py         ->  CER / WER (+ bootstrap CI), segmentation IoU
```

## 4. Installation

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"      # data preparation + tests
pip install -e ".[train]"    # + Kraken / torch (training environment)
pip install -e ".[viz]"      # + Streamlit dashboard
```

## 5. Usage

### Training (local GPU, recommended)

One command fetches the dataset (S3, HuggingFace fallback), fine-tunes on GPU,
and evaluates:

```bash
htmir-train-local --config configs/training.yaml
htmir-train-local --config configs/training.yaml --skip-prepare   # data already present
htmir-train-local --config configs/training.yaml --upload-model   # push model to S3
htmir-train-local --config configs/training.yaml --no-s3          # fully local
```

Set `training.device: cuda:0` in `configs/training.yaml` for the GPU.

### Training (SageMaker, no local GPU)

```bash
python infrastructure/sagemaker_train.py launch --config configs/training.yaml
```
Requires an `ml.g4dn.xlarge` GPU quota on the AWS account.

### Data preparation only

```bash
htmir-prepare --config configs/training.yaml --push-s3
htmir-prepare --config configs/training.yaml --max-files 8 --max-samples 25   # quick sample
```

### Evaluation

```bash
htmir-evaluate --model htmir-french-13c_best.mlmodel \
               --test-arrow data/catmus-french-13c/test.arrow
```

### Dashboard

```bash
streamlit run src/htmir/viz/dashboard.py
```
Also deployed at https://htmirproject.streamlit.app

## 6. Reproducibility

- All runs are driven by `configs/training.yaml` (dataset, model, training,
  SageMaker).
- Random seeds are fixed (e.g. bootstrap resampling).
- Unit tests: `pytest` (data preparation, training command construction,
  evaluation metrics, S3 sync, dashboard).
- Data and model artifacts are versioned on S3.

## 7. Project structure

```
src/htmir/
  data/prepare_catmus.py    extraction CATMuS -> line images + .gt.txt
  data/s3_sync.py           dataset <-> S3 (tar per split)
  training/train_kraken.py  ketos compile + ketos train
  eval/evaluate.py          CER / WER, bootstrap CI
  eval/seg_eval.py          segmentation IoU
  cli/run_local.py          local GPU pipeline (data -> train -> eval)
  viz/dashboard.py          Streamlit dashboard
  utils/logger.py
configs/training.yaml       dataset + model + SageMaker configuration
infrastructure/             SageMaker launcher and container entry point
paper/                      research article (LaTeX) + figures
tests/                      unit tests (pytest)
```

## 8. Research article

**Status: work in progress** — the article is still being finalized.

The deliverable follows the standard research-article structure (abstract,
introduction, related work, methods, results, discussion, conclusion,
bibliography, appendices). Source and auto-generated figures are in `paper/`:

```bash
python paper/make_figures.py        # regenerate figures from metrics
# compile paper/article.tex with a LaTeX toolchain (e.g. Overleaf)
```
