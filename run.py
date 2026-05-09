#!/usr/bin/env python3
"""Bearing fault diagnosis benchmark — CLI entry point.

Examples
--------
    # single combination
    python run.py --model cnn1d --dataset cwru
    python run.py --model svm   --dataset paderborn-real --task multiclass

    # noise-robustness
    python run.py --model cnn1d --dataset cwru --snr 5

    # benchmark every model on a dataset
    python run.py --model all --dataset cwru

    # benchmark every model on every dataset
    python run.py --model all --dataset all
"""

from __future__ import annotations
import argparse
import itertools
import os
import sys
import time

from tqdm.auto import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from bench.config import (
    RunConfig, ALL_DATASETS, ALL_MODELS,
    CLASSICAL_MODELS, CNN_MODELS, DEFAULT_DATA_ROOT, DEFAULT_OUTPUT_DIR,
)
from bench.data import load_dataset
from bench.train import train_classical, train_cnn, RunResult
from bench.utils import (
    set_seed, save_json, device_str, configure_stdout, silence_noisy_warnings,
)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Bearing fault diagnosis benchmark")
    p.add_argument("--model",   required=True,
                   help=f"One of {ALL_MODELS} or 'all'.")
    p.add_argument("--dataset", required=True,
                   help=f"One of {ALL_DATASETS} or 'all'.")
    p.add_argument("--task", default="multiclass",
                   choices=["multiclass", "binary"])
    p.add_argument("--window",  type=int, default=None,
                   help="Window length in samples (default depends on dataset).")
    p.add_argument("--overlap", type=float, default=0.5)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr",      type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--seed",    type=int, default=42)
    p.add_argument("--seeds",   type=int, nargs="+", default=None,
                   help="Run each (model, dataset) once per seed and report "
                        "mean ± std. Overrides --seed when supplied.")
    p.add_argument("--snr",     type=float, default=None,
                   help="If set, AWGN at this SNR (dB) is added to test set.")
    p.add_argument("--data-root",   default=DEFAULT_DATA_ROOT)
    p.add_argument("--output-dir",  default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--device", default="auto",
                   help="'auto', 'cpu', or 'cuda'.")
    p.add_argument("--quiet", action="store_true",
                   help="Suppress per-epoch CNN progress bar.")
    return p.parse_args(argv)


def expand(argval: str, full_list: list[str]) -> list[str]:
    return list(full_list) if argval == "all" else [argval]


def run_one(model: str, dataset: str, args, seed: int,
            show_cnn_progress: bool) -> RunResult:
    cfg = RunConfig(
        model=model, dataset=dataset, task=args.task,
        window=args.window, overlap=args.overlap,
        epochs=args.epochs, batch_size=args.batch_size,
        lr=args.lr, weight_decay=args.weight_decay,
        seed=seed, snr_db=args.snr,
        data_root=os.path.abspath(args.data_root),
        output_dir=os.path.abspath(args.output_dir),
        device=args.device,
    )
    set_seed(cfg.seed)

    ds = load_dataset(
        cfg.dataset, cfg.data_root,
        cfg.resolved_window(), cfg.resolved_step(),
        task=cfg.task,
    )

    if model in CLASSICAL_MODELS:
        result = train_classical(model, ds, cfg, snr_db=cfg.snr_db)
    elif model in CNN_MODELS:
        result = train_cnn(
            model, ds, cfg,
            device=device_str(cfg.device),
            snr_db=cfg.snr_db,
            show_progress=show_cnn_progress,
        )
    else:
        raise ValueError(f"Unknown model '{model}'")

    snr_suffix  = f"_snr{cfg.snr_db}" if cfg.snr_db is not None else ""
    seed_suffix = f"_s{seed}"
    out_path = os.path.join(
        cfg.output_dir,
        f"{cfg.dataset}__{cfg.model}__{cfg.task}{snr_suffix}{seed_suffix}.json",
    )
    save_json(result.to_dict(), out_path)
    return result


def main(argv=None):
    configure_stdout()
    silence_noisy_warnings()
    args = parse_args(argv)

    if args.model not in ALL_MODELS + ["all"]:
        print(f"Unknown --model '{args.model}'. Choose from {ALL_MODELS} or 'all'.")
        sys.exit(2)
    if args.dataset not in ALL_DATASETS + ["all"]:
        print(f"Unknown --dataset '{args.dataset}'. Choose from {ALL_DATASETS} or 'all'.")
        sys.exit(2)

    models   = expand(args.model,   ALL_MODELS)
    datasets = expand(args.dataset, ALL_DATASETS)
    seeds    = args.seeds if args.seeds is not None else [args.seed]
    combos   = list(itertools.product(datasets, models, seeds))

    output_dir = os.path.abspath(args.output_dir)

    summary = []
    t0 = time.perf_counter()
    grid = tqdm(combos, desc="benchmark", unit="run")
    for ds_name, model, seed in grid:
        grid.set_postfix_str(f"{model} / {ds_name} (seed={seed})")
        try:
            res = run_one(model, ds_name, args, seed=seed,
                          show_cnn_progress=not args.quiet)
            summary.append({
                "model":         res.model,
                "dataset":       res.dataset,
                "task":          res.task,
                "seed":          seed,
                "snr":           res.extras.get("snr_db"),
                "accuracy":      res.metrics["accuracy"],
                "f1_macro":      res.metrics["f1_macro"],
                "params":        res.efficiency.get("params"),
                "size_mb":       res.efficiency.get("model_size_mb"),
                "ms_per_window": res.efficiency.get("total_ms_per_window"),
                "train_time_s":  res.train_time_s,
            })
        except FileNotFoundError as e:
            tqdm.write(f"[SKIP] {ds_name}: {e}")
        except Exception as e:
            tqdm.write(f"[ERROR] {model} on {ds_name}: {e}")

    if summary:
        out_path = os.path.join(output_dir, f"summary_{int(time.time())}.json")
        save_json(summary, out_path)
        _print_summary_table(summary, len(seeds) > 1)
        print(f"\nSummary -> {out_path}")
    print(f"Total elapsed: {time.perf_counter() - t0:.1f}s")


def _print_summary_table(summary: list[dict], multi_seed: bool) -> None:
    """Print a tidy summary, aggregating over seeds when multi_seed is True."""
    import statistics
    col_w  = 26

    if not multi_seed:
        header = ("model".ljust(12) + "dataset".ljust(col_w) +
                  "acc".ljust(8) + "f1".ljust(8) +
                  "size_mb".ljust(10) + "ms/win".ljust(10) + "train_s")
        print()
        print(header)
        print("-" * len(header))
        for row in summary:
            print(
                f"{row['model']:<12}"
                f"{row['dataset']:<{col_w}}"
                f"{row['accuracy']:.4f}  "
                f"{row['f1_macro']:.4f}  "
                f"{(row['size_mb'] or 0):>8.2f}  "
                f"{(row['ms_per_window'] or 0):>8.2f}  "
                f"{row['train_time_s']:>6.1f}"
            )
        return

    groups: dict[tuple[str, str], list[dict]] = {}
    for r in summary:
        groups.setdefault((r["model"], r["dataset"]), []).append(r)

    header = ("model".ljust(12) + "dataset".ljust(col_w) +
              "acc (mean±std)".ljust(20) + "f1 (mean±std)".ljust(20) + "n")
    print()
    print(header)
    print("-" * len(header))
    for (model, ds), rows in groups.items():
        accs = [r["accuracy"] for r in rows]
        f1s  = [r["f1_macro"] for r in rows]
        a_m, a_s = statistics.mean(accs), (statistics.stdev(accs) if len(accs) > 1 else 0.0)
        f_m, f_s = statistics.mean(f1s),  (statistics.stdev(f1s)  if len(f1s)  > 1 else 0.0)
        print(
            f"{model:<12}"
            f"{ds:<{col_w}}"
            f"{a_m:.4f} ± {a_s:.4f}    "
            f"{f_m:.4f} ± {f_s:.4f}    "
            f"{len(rows)}"
        )


if __name__ == "__main__":
    main()
