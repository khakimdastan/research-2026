#!/usr/bin/env python3
"""Download and extract the PaderBorn bearing subset used by the benchmark.

The KAt-DataCenter ships each bearing as a single ``.rar`` archive under
https://groups.uni-paderborn.de/kat/BearingDataCenter/ (HTTP).
Each archive contains one ``.mat`` file per (operating condition, run) pair,
named like ``N15_M07_F10_KA01_1.mat``.

This script:
  1. downloads ``.rar`` files for the bearings listed in PADERBORN_BEARINGS,
  2. extracts them into ``data/paderborn/<bearing_id>/`` using either the
     ``rarfile`` Python library (preferred) or the ``unrar`` system tool.
"""

from __future__ import annotations
import os
import shutil
import ssl
import subprocess
import sys
import urllib.request

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from bench.config import DEFAULT_DATA_ROOT
from bench.data import PADERBORN_BEARINGS

from tqdm.auto import tqdm

BASE_URL = "https://groups.uni-paderborn.de/kat/BearingDataCenter"


def _all_bearing_ids() -> list[str]:
    ids = []
    for info in PADERBORN_BEARINGS.values():
        ids.extend(info["ids"])
    return ids


def _remote_size(url: str) -> int | None:
    """HEAD request to get the expected archive size. Returns None on failure."""
    try:
        ctx = ssl.create_default_context()
        req = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(req, context=ctx, timeout=30) as r:
            n = int(r.headers.get("Content-Length", 0))
            return n or None
    except Exception:
        return None


def _download(bid: str, dest_dir: str) -> str | None:
    """Download <bid>.rar to dest_dir with resume + size verification.

    The file is written to ``<bid>.rar.partial`` first and only renamed to its
    final name after the download finishes successfully, so an interrupted run
    can never leave behind a corrupt archive that looks complete.
    """
    url = f"{BASE_URL}/{bid}.rar"
    out = os.path.join(dest_dir, f"{bid}.rar")
    tmp = out + ".partial"
    expected = _remote_size(url)

    if os.path.exists(out):
        if expected is None or os.path.getsize(out) == expected:
            return out
        tqdm.write(f"  [{bid}] cached file size mismatch — re-downloading")
        os.remove(out)
    if os.path.exists(tmp):
        os.remove(tmp)

    try:
        ctx = ssl.create_default_context()
        with urllib.request.urlopen(url, context=ctx, timeout=120) as r:
            total = int(r.headers.get("Content-Length", 0)) or expected
            with open(tmp, "wb") as f, \
                    tqdm(total=total, unit="B", unit_scale=True,
                         desc=f"  {bid}.rar", leave=False) as bar:
                chunk = 1024 * 256
                while True:
                    buf = r.read(chunk)
                    if not buf:
                        break
                    f.write(buf)
                    bar.update(len(buf))
        if expected is not None and os.path.getsize(tmp) != expected:
            raise IOError(
                f"size mismatch (got {os.path.getsize(tmp)}, expected {expected})"
            )
        os.replace(tmp, out)
        return out
    except Exception as e:
        tqdm.write(f"  [{bid}] download FAILED: {e}")
        for p in (tmp, out):
            if os.path.exists(p):
                os.remove(p)
        return None


def _extract(rar_path: str, dest_dir: str, bid: str) -> bool:
    """Try several RAR backends in order; only complain if they all fail.

    Order: bsdtar (built-in on macOS) -> unar -> unrar -> rarfile (python).
    """
    target = os.path.join(dest_dir, bid)
    os.makedirs(target, exist_ok=True)
    attempts: list[str] = []

    def _has_mat() -> bool:
        return any(f.endswith(".mat") for r, _, fs in os.walk(target) for f in fs)

    if shutil.which("bsdtar"):
        try:
            subprocess.check_call(
                ["bsdtar", "-xf", rar_path, "-C", target],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
            if _has_mat():
                return True
            attempts.append("bsdtar produced no .mat files")
        except subprocess.CalledProcessError as e:
            attempts.append(f"bsdtar exit={e.returncode}")

    if shutil.which("unar"):
        try:
            subprocess.check_call(
                ["unar", "-o", target, "-D", rar_path],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
            if _has_mat():
                return True
            attempts.append("unar produced no .mat files")
        except subprocess.CalledProcessError as e:
            attempts.append(f"unar exit={e.returncode}")

    if shutil.which("unrar"):
        try:
            subprocess.check_call(
                ["unrar", "x", "-o+", rar_path, target + os.sep],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
            if _has_mat():
                return True
            attempts.append("unrar produced no .mat files")
        except subprocess.CalledProcessError as e:
            attempts.append(f"unrar exit={e.returncode}")

    try:
        import rarfile
        try:
            with rarfile.RarFile(rar_path) as rf:
                rf.extractall(path=target)
            if _has_mat():
                return True
            attempts.append("rarfile produced no .mat files")
        except Exception as e:
            attempts.append(f"rarfile: {e}")
    except ImportError:
        attempts.append("rarfile not installed")

    tqdm.write(f"  [{bid}] extraction failed (tried: {'; '.join(attempts)})")
    return False


def _flatten(target: str, bid: str) -> None:
    """Move .mat files out of any inner subfolder so that they live directly
    under ``data/paderborn/<bid>/``, then remove empty leftover subfolders.
    """
    for root, _dirs, files in os.walk(target):
        for f in files:
            if f.endswith(".mat"):
                src = os.path.join(root, f)
                dst = os.path.join(target, f)
                if os.path.abspath(src) != os.path.abspath(dst):
                    shutil.move(src, dst)
    for entry in os.listdir(target):
        full = os.path.join(target, entry)
        if os.path.isdir(full):
            try:
                shutil.rmtree(full)
            except Exception:
                pass


def main():
    out_root = os.path.join(DEFAULT_DATA_ROOT, "paderborn")
    os.makedirs(out_root, exist_ok=True)
    cache_dir = os.path.join(out_root, "_rar")
    os.makedirs(cache_dir, exist_ok=True)

    bearings = _all_bearing_ids()
    n_done, n_skip, fails = 0, 0, []
    for bid in tqdm(bearings, desc="PaderBorn download", unit="bearing"):
        target = os.path.join(out_root, bid)
        if os.path.isdir(target) and any(f.endswith(".mat") for f in os.listdir(target)):
            n_skip += 1
            continue
        rar = _download(bid, cache_dir)
        if rar is None:
            fails.append(bid)
            continue
        if _extract(rar, out_root, bid):
            _flatten(os.path.join(out_root, bid), bid)
            n_done += 1
            continue

        # Extraction failed — assume the cached .rar is corrupt and try once more.
        tqdm.write(f"  [{bid}] cached archive looked bad — re-downloading once")
        try:
            os.remove(rar)
        except OSError:
            pass
        rar = _download(bid, cache_dir)
        if rar and _extract(rar, out_root, bid):
            _flatten(os.path.join(out_root, bid), bid)
            n_done += 1
        else:
            fails.append(bid)

    print(f"\nDone. Bearings downloaded+extracted: {n_done}, "
          f"already present: {n_skip}, failed: {len(fails)}.")
    if fails:
        print("\nThe extraction step requires one of:")
        print("  - 'bsdtar'  (already installed on macOS)")
        print("  - 'unar' / 'unrar' (brew install unar  or  brew install unrar)")
        print("  - or simply install the Python package 'rarfile' on a machine with one of the above")
        print(f"Failed bearings: {fails}")
        print(f"Raw .rar archives are cached in: {cache_dir}")
        print("After installing an extractor, re-run this script — it will reuse the cached .rar files.")


if __name__ == "__main__":
    main()
