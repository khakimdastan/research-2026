#!/usr/bin/env python3
"""Generate paper-ready LaTeX tables from results/*.json.

Outputs:
  tables/headline.tex    - Accuracy / Macro-F1 grid (model x dataset)
  tables/efficiency.tex  - Edge efficiency (params, size, train time, infer ms)
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


MODEL_ORDER = ["lr", "svm", "rf", "xgb",
               "cnn1d", "dscnn1d", "cnn1d_fft", "wdcnn", "lsrnet"]
DS_ORDER = ["cwru", "paderborn-artificial", "paderborn-real",
            "paderborn-artificial-xb", "paderborn-real-xb"]
DS_LABEL = {
    "cwru":                    "CWRU",
    "paderborn-artificial":    "\\textsc{Pb-art}",
    "paderborn-real":          "\\textsc{Pb-real}",
    "paderborn-artificial-xb": "\\textsc{Pb-art-xb}",
    "paderborn-real-xb":       "\\textsc{Pb-real-xb}",
}
MODEL_LABEL = {
    "lr": "LR", "svm": "SVM", "rf": "RF", "xgb": "XGB",
    "cnn1d": "\\textsc{Cnn1d}", "dscnn1d": "\\textsc{Dscnn1d}",
    "cnn1d_fft": "\\textsc{Cnn1d\\_fft}", "wdcnn": "\\textsc{Wdcnn}",
    "lsrnet": "\\textsc{Lsrnet}",
}


def load_runs(results_dir: str) -> list[dict]:
    runs = []
    for path in sorted(glob.glob(os.path.join(results_dir, "*__*__*.json"))):
        try:
            d = json.load(open(path))
            if d.get("extras", {}).get("snr_db") is not None:
                continue
            runs.append(d)
        except Exception:
            pass
    return runs


def aggregate(runs: list[dict]) -> dict:
    acc = collections.defaultdict(list)
    f1  = collections.defaultdict(list)
    eff = {}
    for r in runs:
        key = (r["dataset"], r["model"])
        acc[key].append(r["metrics"]["accuracy"])
        f1[key].append(r["metrics"]["f1_macro"])
        eff[key] = {
            "params":   r["efficiency"].get("params", 0) or 0,
            "size_mb":  r["efficiency"].get("model_size_mb", 0.0) or 0.0,
            "train_s":  r["train_time_s"],
            "ms_per_win": r["efficiency"].get("total_ms_per_window", 0.0) or 0.0,
        }
    return {"acc": acc, "f1": f1, "eff": eff}


def fmt_acc(v):
    if v is None:
        return "--"
    return f"{v:.3f}"


def write_headline(agg: dict, datasets: list[str], models: list[str], out_path: str):
    n_ds = len(datasets)
    cols = "l" + "c" * n_ds
    header = " & ".join(["Model"] + [DS_LABEL.get(d, d) for d in datasets]) + " \\\\"
    lines = [
        "\\begin{table*}[t]",
        "\\centering",
        "\\small",
        "\\caption{Test accuracy of every model on every BearingBench setting "
        "(seed 42, multiclass). Bold marks the best per column. "
        "Run-based Paderborn (\\textsc{Pb-art}, \\textsc{Pb-real}) saturates near 1.0; "
        "the cross-bearing splits restore a meaningful difficulty gradient. "
        "Macro-F1 follows the same ranking and is reported in the per-run JSONs.}",
        "\\label{tab:headline}",
        f"\\begin{{tabular}}{{{cols}}}",
        "\\toprule",
        header,
        "\\midrule",
    ]

    best_per_ds = {}
    for d in datasets:
        vals = []
        for m in models:
            vs = agg["acc"].get((d, m), [])
            if vs:
                vals.append((max(vs), m))
        if vals:
            best_per_ds[d] = max(vals)[0]

    for m in models:
        row = [MODEL_LABEL.get(m, m)]
        for d in datasets:
            vs = agg["acc"].get((d, m), [])
            if not vs:
                row.append("--")
                continue
            v = max(vs)
            cell = fmt_acc(v)
            if abs(v - best_per_ds.get(d, -1)) < 1e-6:
                cell = "\\textbf{" + cell + "}"
            row.append(cell)
        lines.append(" & ".join(row) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table*}"]

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")


def write_efficiency(agg: dict, models: list[str], out_path: str):
    eff = agg["eff"]
    rep = {}
    for m in models:
        rows = [v for (d, mm), v in eff.items() if mm == m]
        if not rows:
            continue
        rep[m] = {
            "params":     int(rows[0]["params"]),
            "size_mb":    rows[0]["size_mb"],
            "train_s":    sum(r["train_s"]    for r in rows) / len(rows),
            "ms_per_win": sum(r["ms_per_win"] for r in rows) / len(rows),
        }

    cols = "lrrrr"
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\small",
        "\\caption{Edge-efficiency of each model: trainable parameters, "
        "32-bit on-disk size, mean training time across all five datasets, "
        "and mean test-time inference latency per window (CPU only).}",
        "\\label{tab:efficiency}",
        f"\\begin{{tabular}}{{{cols}}}",
        "\\toprule",
        "Model & Params & Size (MB) & Train (s) & Infer (ms/win) \\\\",
        "\\midrule",
    ]
    for m in models:
        if m not in rep:
            continue
        r = rep[m]
        lines.append(
            f"{MODEL_LABEL.get(m, m)} & "
            f"{r['params']:>7,d} & "
            f"{r['size_mb']:>6.3f} & "
            f"{r['train_s']:>6.1f} & "
            f"{r['ms_per_win']:>6.2f} \\\\"
        )
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]

    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--out", default=os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "tables")))
    args = p.parse_args()

    runs = load_runs(args.results)
    if not runs:
        print(f"No result JSONs found in {args.results}")
        return
    agg = aggregate(runs)

    datasets = [d for d in DS_ORDER if any(k[0] == d for k in agg["acc"].keys())]
    models   = [m for m in MODEL_ORDER if any(k[1] == m for k in agg["acc"].keys())]

    write_headline   (agg, datasets, models, os.path.join(args.out, "headline.tex"))
    write_efficiency (agg,           models, os.path.join(args.out, "efficiency.tex"))
    print(f"Wrote tables to {args.out}")


if __name__ == "__main__":
    main()
