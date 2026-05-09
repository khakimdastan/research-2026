# BearingBench

A unified, extensible benchmark for **classical ML** and **lightweight 1D
deep learning** on rolling bearing fault diagnosis.

## Datasets

| Name                       | Protocol      | Source                                          |
|----------------------------|---------------|-------------------------------------------------|
| `cwru`                     | leave-one-load-out | CWRU drive-end vibration @ 12 kHz          |
| `paderborn-artificial`     | run-based     | Paderborn KAt-DataCenter, EDM/drilled @ 64 kHz  |
| `paderborn-real`           | run-based     | Paderborn KAt-DataCenter, real-fatigue @ 64 kHz |
| `paderborn-artificial-xb`  | cross-bearing | same source, holds out unseen physical bearings |
| `paderborn-real-xb`        | cross-bearing | same source, holds out unseen physical bearings |

All splits are at the **recording-file level**. The `-xb` variants additionally
hold out entire physical bearings from the test set.

## Models

- Classical (on 22 hand-crafted features): `lr`, `svm`, `rf`, `xgb`
- Lightweight 1D CNNs: `cnn1d`, `dscnn1d`, `cnn1d_fft`, `wdcnn`, `lsrnet`

## Repository skeleton

```
bearing_bench/
├── README.md
├── requirements.txt
├── run.py                            CLI entry point
├── bench/                            core library
│   ├── config.py
│   ├── data.py
│   ├── features.py
│   ├── models_classical.py
│   ├── models_cnn.py
│   ├── train.py
│   ├── metrics.py
│   ├── noise.py
│   └── utils.py
├── scripts/
│   ├── download_cwru.py
│   ├── download_paderborn.py
│   └── run_robustness.py
├── data/                             raw .mat files (gitignored)
└── results/                          per-run JSON metrics + summary
```

## Setup

### 1. Create an environment and install requirements

```bash
cd bearing_bench
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Install a RAR extractor (PaderBorn only)

PaderBorn ships each bearing as a `.rar` archive, so you need one of the
following on your `PATH`:

| Platform | Easiest option                                   |
|----------|--------------------------------------------------|
| macOS    | `bsdtar` (already installed)                     |
| macOS    | `brew install unar` _(alternative)_              |
| Linux    | `sudo apt install unrar` or `sudo apt install unar` |
| Windows  | install [WinRAR](https://www.win-rar.com/) and add `unrar.exe` to `PATH` |

You do **not** need this if you only run CWRU.

### 3. Download the data

```bash
python scripts/download_cwru.py        # ~50 MB
python scripts/download_paderborn.py   # ~1.8 GB, will use the RAR extractor
```

Both scripts show a `tqdm` progress bar. They skip files that are already
present, so you can stop and resume safely.

After they finish your `data/` folder will look like:

```
data/
├── cwru/                 97.mat, 98.mat, ...
└── paderborn/
    ├── K001/             N15_M07_F10_K001_1.mat, N15_M07_F10_K001_2.mat, ...
    ├── KA01/
    └── ...
```

### 4. Sanity-check the pipeline

```bash
python run.py --model rf --dataset cwru
```

That should produce a results file under `results/` and a one-line summary
table.

## Running benchmarks

The CLI takes a `--model` and a `--dataset` flag. Use `all` for either to
sweep across the full grid.

```bash
# Single (model, dataset) pair
python run.py --model cnn1d --dataset paderborn-artificial

# Every model on one dataset
python run.py --model all --dataset cwru

# Full benchmark grid (every model × every dataset)
python run.py --model all --dataset all

# Binary task (Healthy vs Faulty)
python run.py --model svm --dataset cwru --task binary

# Test-set noise robustness
python run.py --model cnn1d --dataset cwru --snr 5

# Multi-SNR sweep
python scripts/run_robustness.py --model all --dataset cwru \
    --snr-list 20 10 5 0 -5
```

A multi-run invocation prints a final summary table and writes
`results/summary_<timestamp>.json` next to the per-run files.

## CLI flags (`run.py`)

| Flag           | Description                                                | Default          |
|----------------|------------------------------------------------------------|------------------|
| `--model`      | `lr` / `svm` / `rf` / `xgb` / `cnn1d` / `dscnn1d` / `all`  | required         |
| `--dataset`    | `cwru` / `paderborn-artificial` / `paderborn-real` / `all` | required         |
| `--task`       | `multiclass` or `binary`                                   | `multiclass`     |
| `--window`     | window length in samples                                   | dataset default  |
| `--overlap`    | window overlap fraction                                    | `0.5`            |
| `--epochs`     | CNN training epochs                                        | `30`             |
| `--batch-size` | CNN batch size                                             | `256`            |
| `--lr`         | CNN Adam learning rate                                     | `1e-3`           |
| `--snr`        | If set, AWGN at this SNR (dB) is added to the test windows | _none_           |
| `--data-root`  | Root folder containing `cwru/` and `paderborn/`            | `./data`         |
| `--output-dir` | JSON results directory                                     | `./results`      |
| `--device`     | `auto`, `cpu`, or `cuda`                                   | `auto`           |

## Outputs per run

Each run writes one JSON file containing:

- **Metrics**: accuracy, macro F1, macro precision/recall, full confusion matrix, `classification_report`.
- **Edge efficiency**:
  - Classical: model size on disk (MB), feature-extraction time, prediction time, total ms/window.
  - CNN: parameter count, model size, forward time per window.
- **Training time** (seconds), `RunConfig` (window/SNR/task), and CNN training history.

## Citing the data

- Smith, W. A., & Randall, R. B. *Rolling element bearing diagnostics using the Case Western Reserve University data: A benchmark study.* MSSP, 2015.
- Lessmeier, C. et al. *Condition monitoring of bearing damage in electromechanical drive systems by using motor current signals of electric motors: A benchmark data set for data-driven classification.* PHM Society European Conference, 2016.
