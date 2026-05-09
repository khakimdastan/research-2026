#!/usr/bin/env python3
"""Sweep noise robustness for one or many (model, dataset) pairs.

For every (model, dataset) the model is trained ONCE on clean training data,
then evaluated at every requested SNR with AWGN added to the test windows.

Example
-------
    python scripts/run_robustness.py --model all --dataset cwru \
        --snr-list 20 10 5 0 -5
"""

from __future__ import annotations
import argparse
import itertools
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from bench.config import (
    ALL_MODELS, ALL_DATASETS, CLASSICAL_MODELS,
    DEFAULT_DATA_ROOT, DEFAULT_OUTPUT_DIR, RunConfig,
)
from bench.data import load_dataset
from bench.train import train_classical, train_cnn
from bench.utils import (
    set_seed, save_json, device_str, configure_stdout, silence_noisy_warnings,
)
from tqdm.auto import tqdm


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model",   required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--task", default="multiclass", choices=["multiclass", "binary"])
    p.add_argument("--snr-list", nargs="+", type=float, default=[20, 10, 5, 0, -5])
    p.add_argument("--epochs",  type=int, default=20)
    p.add_argument("--seed",    type=int, default=42)
    p.add_argument("--data-root",  default=DEFAULT_DATA_ROOT)
    p.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--device", default="auto")
    return p.parse_args()


def expand(arg, full):
    return list(full) if arg == "all" else [arg]


def main():
    configure_stdout()
    silence_noisy_warnings()

    args = parse_args()
    models   = expand(args.model,   ALL_MODELS)
    datasets = expand(args.dataset, ALL_DATASETS)
    combos   = list(itertools.product(datasets, models, args.snr_list))

    rows = []
    grid = tqdm(combos, desc="robustness", unit="run")
    for ds_name, model, snr in grid:
        grid.set_postfix_str(f"{model} / {ds_name} @ {snr} dB")
        cfg = RunConfig(
            model=model, dataset=ds_name, task=args.task,
            epochs=args.epochs, seed=args.seed,
            data_root=os.path.abspath(args.data_root),
            output_dir=os.path.abspath(args.output_dir),
            device=args.device,
        )
        set_seed(cfg.seed)
        try:
            ds = load_dataset(ds_name, cfg.data_root,
                              cfg.resolved_window(), cfg.resolved_step(),
                              task=cfg.task)
            if model in CLASSICAL_MODELS:
                res = train_classical(model, ds, cfg, snr_db=snr)
            else:
                res = train_cnn(model, ds, cfg,
                                 device=device_str(cfg.device),
                                 snr_db=snr, show_progress=False)
        except FileNotFoundError as e:
            tqdm.write(f"[SKIP] {ds_name}: {e}")
            continue
        except Exception as e:
            tqdm.write(f"[ERROR] {model} on {ds_name} @ {snr}: {e}")
            continue

        rows.append({
            "model": model, "dataset": ds_name, "task": args.task,
            "snr_db": snr,
            "accuracy": res.metrics["accuracy"],
            "f1_macro": res.metrics["f1_macro"],
        })

    out_path = os.path.join(
        os.path.abspath(args.output_dir),
        f"robustness_{int(time.time())}.json",
    )
    save_json(rows, out_path)

    print()
    print("ROBUSTNESS SWEEP")
    print("-" * 70)
    for r in rows:
        print(f"{r['model']:<10}{r['dataset']:<25} SNR={r['snr_db']:>5}  "
              f"acc={r['accuracy']:.4f}  f1={r['f1_macro']:.4f}")
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
