#!/usr/bin/env python3
"""Generate paper-ready figures from results/*.json.

Outputs to ``results/figures/``:
  - tradeoff.png        : accuracy vs train-time and accuracy vs model size
  - difficulty.png      : per-dataset accuracy bars per model
  - cwru_severity.png   : CWRU per-class accuracy split by fault size
  - loss_curves.png     : CNN train/val loss curves for each (model, dataset)
  - noise_robustness.png: accuracy vs SNR if results/robustness_*.json exists
"""

from __future__ import annotations
import argparse
import collections
import glob
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from bench.config import DEFAULT_OUTPUT_DIR

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


CLASSICAL = {"lr", "svm", "rf", "xgb"}
CNN       = {"cnn1d", "dscnn1d", "cnn1d_fft"}
COLOR     = {
    "lr": "#1f77b4", "svm": "#ff7f0e", "rf": "#2ca02c", "xgb": "#d62728",
    "cnn1d": "#9467bd", "dscnn1d": "#8c564b", "cnn1d_fft": "#e377c2",
}
MARKER = {m: ("o" if m in CLASSICAL else "s") for m in (CLASSICAL | CNN)}


def _load_runs(results_dir: str) -> list[dict]:
    runs = []
    for path in sorted(glob.glob(os.path.join(results_dir, "*__*__*.json"))):
        try:
            with open(path) as f:
                d = json.load(f)
            if "metrics" not in d or "model" not in d:
                continue
            runs.append(d)
        except Exception:
            pass
    return runs


def _aggregate_by_dataset_model(runs: list[dict]) -> dict:
    acc, f1 = collections.defaultdict(list), collections.defaultdict(list)
    size, ms, train_s = {}, {}, {}
    for r in runs:
        if r.get("extras", {}).get("snr_db") is not None:
            continue
        key = (r["dataset"], r["model"])
        acc[key].append(r["metrics"]["accuracy"])
        f1[key].append(r["metrics"]["f1_macro"])
        size[key]    = r["efficiency"].get("model_size_mb", 0.0)
        ms[key]      = r["efficiency"].get("total_ms_per_window", 0.0)
        train_s[key] = r["train_time_s"]
    return {
        "acc": acc, "f1": f1,
        "size": size, "ms": ms, "train_s": train_s,
    }


def fig_difficulty(agg: dict, out_path: str) -> None:
    datasets = sorted({k[0] for k in agg["acc"].keys()})
    models   = sorted({k[1] for k in agg["acc"].keys()})
    fig, ax = plt.subplots(figsize=(max(8, 1.0 * len(datasets) * len(models) / 4), 4.5))
    width = 0.8 / max(1, len(models))
    x = np.arange(len(datasets))
    for i, m in enumerate(models):
        means = [np.mean(agg["acc"].get((d, m), [np.nan])) for d in datasets]
        stds  = [np.std(agg["acc"].get((d, m), [0]), ddof=0) for d in datasets]
        ax.bar(x + i * width, means, width, yerr=stds, capsize=2,
               label=m, color=COLOR.get(m, "#888"))
    ax.set_xticks(x + width * (len(models) - 1) / 2)
    ax.set_xticklabels(datasets, rotation=20, ha="right")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0.5, 1.02)
    ax.set_title("Accuracy by dataset and model")
    ax.legend(ncol=min(len(models), 4), fontsize=8, loc="lower right")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def fig_tradeoff(agg: dict, out_path: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for (d, m), accs in agg["acc"].items():
        a = float(np.mean(accs))
        axes[0].scatter(agg["train_s"][(d, m)] + 0.01, a,
                         marker=MARKER.get(m, "o"),
                         color=COLOR.get(m, "#888"),
                         s=70, alpha=0.85, edgecolor="black", linewidth=0.4)
        axes[0].annotate(f"{m}/{d.split('-')[0]}",
                          (agg["train_s"][(d, m)] + 0.01, a),
                          fontsize=6, alpha=0.6, xytext=(3, 2),
                          textcoords="offset points")
        axes[1].scatter(agg["size"][(d, m)] + 0.001, a,
                         marker=MARKER.get(m, "o"),
                         color=COLOR.get(m, "#888"),
                         s=70, alpha=0.85, edgecolor="black", linewidth=0.4)
        axes[1].annotate(f"{m}/{d.split('-')[0]}",
                          (agg["size"][(d, m)] + 0.001, a),
                          fontsize=6, alpha=0.6, xytext=(3, 2),
                          textcoords="offset points")
    for ax in axes:
        ax.set_ylabel("Accuracy")
        ax.set_ylim(0.5, 1.02)
        ax.grid(True, alpha=0.3)
    axes[0].set_xscale("log"); axes[0].set_xlabel("Training time (s, log)")
    axes[1].set_xscale("log"); axes[1].set_xlabel("Model size (MB, log)")
    axes[0].set_title("Accuracy vs training cost")
    axes[1].set_title("Accuracy vs model size")
    handles = [plt.Line2D([], [], marker=MARKER.get(m, "o"), linestyle="",
                           color=COLOR.get(m, "#888"), label=m)
               for m in sorted({k[1] for k in agg["acc"].keys()})]
    axes[1].legend(handles=handles, fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def fig_cwru_severity(runs: list[dict], out_path: str) -> None:
    """Per-class CWRU accuracy. Faults are B/IR/OR; we don't have per-severity
    info encoded in the labels (only one per fault type), so we report per-class
    recall instead, which is the closest proxy."""
    rows = []
    for r in runs:
        if r["dataset"] != "cwru" or r["task"] != "multiclass":
            continue
        if r.get("extras", {}).get("snr_db") is not None:
            continue
        rep = r["metrics"]["report"]
        for cls, info in rep.items():
            if cls in {"accuracy", "macro avg", "weighted avg"}:
                continue
            rows.append((r["model"], cls, info["recall"]))
    if not rows:
        return
    models = sorted({r[0] for r in rows})
    classes = sorted({r[1] for r in rows})
    M = np.full((len(models), len(classes)), np.nan)
    for m, c, v in rows:
        i, j = models.index(m), classes.index(c)
        M[i, j] = v if np.isnan(M[i, j]) else (M[i, j] + v) / 2  # avg over seeds
    fig, ax = plt.subplots(figsize=(1.2 * len(classes) + 2, 0.5 * len(models) + 2))
    im = ax.imshow(M, cmap="viridis", vmin=0.5, vmax=1.0)
    ax.set_xticks(range(len(classes))); ax.set_xticklabels(classes)
    ax.set_yticks(range(len(models)));  ax.set_yticklabels(models)
    for i in range(len(models)):
        for j in range(len(classes)):
            v = M[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    color=("white" if v < 0.8 else "black"), fontsize=9)
    fig.colorbar(im, ax=ax, label="recall")
    ax.set_title("CWRU per-class recall")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def fig_loss_curves(runs: list[dict], out_path: str) -> None:
    cnn_runs = [r for r in runs
                if r.get("extras", {}).get("history") is not None
                and r.get("extras", {}).get("snr_db") is None]
    if not cnn_runs:
        return
    by_dataset = collections.defaultdict(list)
    for r in cnn_runs:
        by_dataset[r["dataset"]].append(r)
    n = len(by_dataset)
    fig, axes = plt.subplots(1, n, figsize=(4 * n + 1, 4), sharey=True)
    if n == 1:
        axes = [axes]
    for ax, (d, rs) in zip(axes, sorted(by_dataset.items())):
        seen_models = set()
        for r in rs:
            m = r["model"]
            if m in seen_models:
                continue
            seen_models.add(m)
            h = r["extras"]["history"]
            ep = range(1, len(h["train_loss"]) + 1)
            ax.plot(ep, h["train_loss"],
                    color=COLOR.get(m, "#888"),
                    linestyle="-", alpha=0.9, label=f"{m} train")
            ax.plot(ep, h["val_loss"],
                    color=COLOR.get(m, "#888"),
                    linestyle="--", alpha=0.7, label=f"{m} val")
        ax.set_title(d)
        ax.set_xlabel("Epoch")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7)
    axes[0].set_ylabel("Loss")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def fig_noise_robustness(results_dir: str, out_path: str) -> None:
    files = sorted(glob.glob(os.path.join(results_dir, "robustness_*.json")))
    if not files:
        return
    rows = []
    for f in files:
        with open(f) as fh:
            rows.extend(json.load(fh))
    if not rows:
        return
    by = collections.defaultdict(list)
    for r in rows:
        by[(r["dataset"], r["model"])].append((r["snr_db"], r["accuracy"]))
    datasets = sorted({k[0] for k in by})
    fig, axes = plt.subplots(1, len(datasets), figsize=(5 * len(datasets) + 1, 4),
                              sharey=True)
    if len(datasets) == 1:
        axes = [axes]
    for ax, d in zip(axes, datasets):
        for (ds, m), pts in by.items():
            if ds != d:
                continue
            pts_sorted = sorted(pts)
            xs = [p[0] for p in pts_sorted]
            ys = [p[1] for p in pts_sorted]
            ax.plot(xs, ys, marker=MARKER.get(m, "o"),
                    color=COLOR.get(m, "#888"), label=m)
        ax.set_title(d); ax.set_xlabel("SNR (dB)")
        ax.grid(True, alpha=0.3)
        ax.invert_xaxis()
        ax.set_ylim(0.0, 1.02)
    axes[0].set_ylabel("Accuracy")
    axes[-1].legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", default=DEFAULT_OUTPUT_DIR)
    args = p.parse_args()
    out_dir = os.path.join(args.results, "figures")
    os.makedirs(out_dir, exist_ok=True)

    runs = _load_runs(args.results)
    if not runs:
        print(f"No result JSONs found in {args.results}")
        return
    agg = _aggregate_by_dataset_model(runs)

    fig_difficulty (agg,  os.path.join(out_dir, "difficulty.png"))
    fig_tradeoff   (agg,  os.path.join(out_dir, "tradeoff.png"))
    fig_cwru_severity(runs, os.path.join(out_dir, "cwru_severity.png"))
    fig_loss_curves(runs, os.path.join(out_dir, "loss_curves.png"))
    fig_noise_robustness(args.results, os.path.join(out_dir, "noise_robustness.png"))

    print(f"Figures written to {out_dir}")


if __name__ == "__main__":
    main()
