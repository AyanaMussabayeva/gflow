from __future__ import annotations

import json
from pathlib import Path

from pkg.config import CONFIGS_ROOT


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_profile_config(profile_name: str) -> dict:
    path = CONFIGS_ROOT / "profiles" / f"{profile_name}.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing profile config: {path}")
    return _load_json(path)


def load_benchmark_config(benchmark_name: str) -> dict:
    path = CONFIGS_ROOT / "benchmarks" / f"{benchmark_name}.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing benchmark config: {path}")
    return _load_json(path)


def load_suite_config(suite_name: str) -> dict:
    path = CONFIGS_ROOT / "suites" / f"{suite_name}.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing suite config: {path}")
    return _load_json(path)
