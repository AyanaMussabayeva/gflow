from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

from pkg.config import ARTIFACTS_V2_ROOT, REPORT_ASSETS_V2_ROOT
from pkg.reporting.artifacts import _json_default
from pkg.reporting.figures import (
    plot_diagnostics,
    plot_implementation_comparison,
    plot_regret_comparison,
    plot_reward_histogram,
)


@dataclass
class ReportAssetBundle:
    root: Path
    img_root: Path
    comparison_root: Path
    tables_root: Path
    snippets_root: Path

    @classmethod
    def create(cls, suite_name: str) -> "ReportAssetBundle":
        root = REPORT_ASSETS_V2_ROOT / suite_name / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        return cls.from_root(root)

    @classmethod
    def from_root(cls, root: Path) -> "ReportAssetBundle":
        img_root = root / "img"
        comparison_root = root / "comparison"
        tables_root = root / "tables"
        snippets_root = root / "snippets"
        img_root.mkdir(parents=True, exist_ok=True)
        comparison_root.mkdir(parents=True, exist_ok=True)
        tables_root.mkdir(parents=True, exist_ok=True)
        snippets_root.mkdir(parents=True, exist_ok=True)
        return cls(
            root=root,
            img_root=img_root,
            comparison_root=comparison_root,
            tables_root=tables_root,
            snippets_root=snippets_root,
        )

    def for_spec(self, spec_name: str) -> "ReportAssetBundle":
        return self.from_root(self.root / spec_name)

    def write_manifest(self, payload: dict) -> Path:
        path = self.root / "manifest.json"
        path.write_text(json.dumps(payload, indent=2, default=_json_default) + "\n", encoding="utf-8")
        return path


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def latest_suite_manifest(suite_name: str) -> Path:
    candidates = sorted((ARTIFACTS_V2_ROOT / "_suites" / suite_name).glob("*/manifest.json"))
    if not candidates:
        raise FileNotFoundError(f"No suite manifest found for suite={suite_name}")
    return candidates[-1]


def load_suite_manifest(suite_name: str, suite_manifest_path: str | None = None) -> tuple[Path, dict]:
    manifest_path = latest_suite_manifest(suite_name) if suite_manifest_path is None else Path(suite_manifest_path)
    return manifest_path, _load_json(manifest_path)


def resolve_artifact_dir_from_suite_manifest(
    spec_name: str,
    benchmark_name: str,
    suite_name: str,
    suite_manifest_path: str | None = None,
) -> Path:
    _, manifest = load_suite_manifest(suite_name, suite_manifest_path=suite_manifest_path)
    for run in manifest.get("runs", []):
        if run.get("spec_name") == spec_name and run.get("benchmark") == benchmark_name:
            return Path(run["artifact_dir"])
    raise FileNotFoundError(
        f"No matching run in suite manifest for suite={suite_name} spec={spec_name} benchmark={benchmark_name}"
    )


def _artifact_summary_path(
    spec_name: str,
    benchmark_name: str,
    artifact_dir: str | None = None,
    suite_name: str | None = None,
    suite_manifest_path: str | None = None,
) -> Path:
    if artifact_dir is not None:
        return Path(artifact_dir) / "summary.json"
    if suite_name is not None:
        return resolve_artifact_dir_from_suite_manifest(
            spec_name=spec_name,
            benchmark_name=benchmark_name,
            suite_name=suite_name,
            suite_manifest_path=suite_manifest_path,
        ) / "summary.json"
    candidates = sorted((ARTIFACTS_V2_ROOT / spec_name / benchmark_name).glob("*/summary.json"))
    if not candidates:
        raise FileNotFoundError(f"No artifact found for spec={spec_name} benchmark={benchmark_name}")
    return candidates[-1]


def load_v2_artifact(
    spec_name: str,
    benchmark_name: str,
    artifact_dir: str | None = None,
    suite_name: str | None = None,
    suite_manifest_path: str | None = None,
) -> tuple[Path, dict]:
    summary_path = _artifact_summary_path(
        spec_name,
        benchmark_name,
        artifact_dir,
        suite_name=suite_name,
        suite_manifest_path=suite_manifest_path,
    )
    return summary_path.parent, _load_json(summary_path)


def load_legacy_benchmark_results(benchmark_name: str) -> dict:
    path = Path("legacy") / "results" / "compute_tradeoff" / f"{benchmark_name}_full" / "raw_results.json"
    data = _load_json(path)
    return data[benchmark_name]


def _trial_final_regret(trial: dict) -> float:
    return float(trial["regrets"][-1])


def select_representative_trial(trials: list[dict]) -> dict:
    ordered = sorted(trials, key=_trial_final_regret)
    return ordered[len(ordered) // 2]


def export_report_like_figures(
    spec_name: str,
    benchmark_name: str,
    artifact_dir: str | None = None,
    suite_name: str = "main_v2",
    suite_manifest_path: str | None = None,
    bundle: ReportAssetBundle | None = None,
) -> dict:
    artifact_root, payload = load_v2_artifact(
        spec_name,
        benchmark_name,
        artifact_dir=artifact_dir,
        suite_name=suite_name,
        suite_manifest_path=suite_manifest_path,
    )
    bundle = ReportAssetBundle.create(suite_name) if bundle is None else bundle
    summary = payload["summary"]
    random_trial = select_representative_trial(payload["random_trials"])
    gfn_trial = select_representative_trial(payload["gfn_trials"])

    regret_path = bundle.img_root / f"{benchmark_name}_regret_comparison.png"
    plot_regret_comparison(
        summary_by_method=summary,
        output_path=str(regret_path),
        title=f"{benchmark_name}: v2 random vs gfn",
    )

    random_diag_path = bundle.img_root / f"{benchmark_name}_random_diagnostics.png"
    gfn_diag_path = bundle.img_root / f"{benchmark_name}_gfn_diagnostics.png"
    plot_diagnostics(random_trial, str(random_diag_path), f"{benchmark_name}: representative random seed")
    plot_diagnostics(gfn_trial, str(gfn_diag_path), f"{benchmark_name}: representative gfn seed")

    reward_hist_path = bundle.img_root / f"{benchmark_name}_reward_hist.png"
    plot_reward_histogram(
        random_trials=payload["random_trials"],
        gfn_trials=payload["gfn_trials"],
        output_path=str(reward_hist_path),
        title=f"{benchmark_name}: reward distribution",
    )

    legacy_results = load_legacy_benchmark_results(benchmark_name)
    comparison_path = bundle.comparison_root / f"{benchmark_name}_legacy_vs_v2_regret.png"
    plot_implementation_comparison(
        legacy_results=legacy_results,
        v2_summary=summary,
        output_path=str(comparison_path),
        title=f"{benchmark_name}: legacy vs v2",
    )

    return {
        "artifact_root": str(artifact_root),
        "regret_comparison": str(regret_path),
        "regret_alias": str(bundle.img_root / f"{benchmark_name}_regret.png") if benchmark_name == "branin" else None,
        "random_diagnostics": str(random_diag_path),
        "gfn_diagnostics": str(gfn_diag_path),
        "reward_hist": str(reward_hist_path),
        "legacy_vs_v2": str(comparison_path),
    }
