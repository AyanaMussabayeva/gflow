from __future__ import annotations

import numpy as np

from pkg.benchmarks import Benchmark
from pkg.rewards import StepContext, build_step_context
from pkg.surrogates import Standardizer


def dataset_context(
    benchmark: Benchmark,
    x: np.ndarray,
    y: np.ndarray,
    standardizer: Standardizer,
    n_init: int,
    n_iter: int,
) -> np.ndarray:
    x_norm = standardizer.normalize_x(x)
    y_norm = standardizer.normalize_y(y)
    progress = (len(y) - n_init) / max(n_iter, 1)
    return np.array(
        [
            float(progress),
            float(np.max(y_norm)),
            float(np.mean(y_norm)),
            float(np.std(y_norm)),
            float(np.mean(np.var(x_norm, axis=0))),
            benchmark.dim / 10.0,
        ],
        dtype=np.float32,
    )


__all__ = ["StepContext", "build_step_context", "dataset_context"]
