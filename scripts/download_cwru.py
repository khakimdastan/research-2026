#!/usr/bin/env python3
"""Download the CWRU subset used by the benchmark.

Files are saved to data/cwru/ relative to the repo root.
"""

from __future__ import annotations
import os
import shutil
import ssl
import sys
import urllib.request

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from bench.config import DEFAULT_DATA_ROOT
from bench.data import CWRU_FILES

from tqdm.auto import tqdm

BASE_URL = "https://engineering.case.edu/sites/default/files/"


def _download_with_progress(url: str, dest: str) -> None:
    ctx = ssl.create_default_context()
    with urllib.request.urlopen(url, context=ctx, timeout=60) as r:
        total = int(r.headers.get("Content-Length", 0)) or None
        with open(dest, "wb") as f, \
                tqdm(total=total, unit="B", unit_scale=True,
                     desc=os.path.basename(dest), leave=False) as bar:
            chunk = 1024 * 64
            while True:
                buf = r.read(chunk)
                if not buf:
                    break
                f.write(buf)
                bar.update(len(buf))


def main():
    out_dir = os.path.join(DEFAULT_DATA_ROOT, "cwru")
    os.makedirs(out_dir, exist_ok=True)
    n_ok, n_skip, n_fail = 0, 0, 0
    todo = [(f, *r) for (f, *r) in CWRU_FILES
            if not os.path.exists(os.path.join(out_dir, f))]
    n_skip = len(CWRU_FILES) - len(todo)
    for fname, *_ in tqdm(todo, desc="CWRU download", unit="file"):
        dest = os.path.join(out_dir, fname)
        url = BASE_URL + fname
        try:
            _download_with_progress(url, dest)
            n_ok += 1
        except Exception as e:
            tqdm.write(f"  FAILED {fname}: {e}")
            if os.path.exists(dest):
                os.remove(dest)
            n_fail += 1
    print(f"\nDone. New: {n_ok}, already present: {n_skip}, "
          f"failed: {n_fail}, dir: {out_dir}")


if __name__ == "__main__":
    main()
