"""Global configuration: paths, dataset registry, defaults."""

from __future__ import annotations
import os
from dataclasses import dataclass, field

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEFAULT_DATA_ROOT = os.path.join(REPO_ROOT, "data")
DEFAULT_OUTPUT_DIR = os.path.join(REPO_ROOT, "results")

CWRU_FS = 12_000
PADERBORN_FS = 64_000


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    fs: int
    default_window: int
    default_overlap: float = 0.5


DATASET_REGISTRY: dict[str, DatasetSpec] = {
    "cwru":                    DatasetSpec("cwru",                    CWRU_FS,      1024),
    "paderborn-artificial":    DatasetSpec("paderborn-artificial",    PADERBORN_FS, 4096),
    "paderborn-real":          DatasetSpec("paderborn-real",          PADERBORN_FS, 4096),
    "paderborn-artificial-xb": DatasetSpec("paderborn-artificial-xb", PADERBORN_FS, 4096),
    "paderborn-real-xb":       DatasetSpec("paderborn-real-xb",       PADERBORN_FS, 4096),
}

ALL_DATASETS = list(DATASET_REGISTRY.keys())

CLASSICAL_MODELS = ["lr", "svm", "rf", "xgb"]
CNN_MODELS       = ["cnn1d", "dscnn1d", "cnn1d_fft", "wdcnn", "lsrnet"]
ALL_MODELS       = CLASSICAL_MODELS + CNN_MODELS


@dataclass
class RunConfig:
    model: str
    dataset: str
    task: str = "multiclass"
    window: int | None = None
    overlap: float = 0.5
    epochs: int = 30
    batch_size: int = 256
    lr: float = 1e-3
    weight_decay: float = 1e-4
    seed: int = 42
    snr_db: float | None = None
    data_root: str = DEFAULT_DATA_ROOT
    output_dir: str = DEFAULT_OUTPUT_DIR
    device: str = "auto"

    def resolved_window(self) -> int:
        if self.window is not None:
            return self.window
        return DATASET_REGISTRY[self.dataset].default_window

    def resolved_step(self) -> int:
        return max(1, int(self.resolved_window() * (1.0 - self.overlap)))
