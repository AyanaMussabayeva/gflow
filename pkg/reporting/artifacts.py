from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

from pkg.config import ARTIFACTS_V2_ROOT


def _json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    raise TypeError(f"Unsupported type: {type(value)!r}")


@dataclass
class ArtifactRun:
    root: Path
    figures_root: Path
    trials_root: Path
    trial_figures_root: Path

    @classmethod
    def create(cls, spec_name: str, benchmark_name: str) -> "ArtifactRun":
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        root = ARTIFACTS_V2_ROOT / spec_name / benchmark_name / timestamp
        root.mkdir(parents=True, exist_ok=False)
        figures_root = root / "figures"
        trials_root = root / "trials"
        trial_figures_root = figures_root / "trials"
        figures_root.mkdir()
        trials_root.mkdir()
        trial_figures_root.mkdir(parents=True)
        return cls(
            root=root,
            figures_root=figures_root,
            trials_root=trials_root,
            trial_figures_root=trial_figures_root,
        )

    def write_json(self, name: str, payload: dict) -> Path:
        path = self.root / name
        self.write_json_to_path(path, payload)
        return path

    def write_json_to_path(self, path: Path, payload: dict) -> Path:
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        tmp_path.write_text(json.dumps(payload, indent=2, default=_json_default) + "\n", encoding="utf-8")
        tmp_path.replace(path)
        return path
