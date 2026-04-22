from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pkg.experiments.suites import BENCHMARKS
from pkg.reporting import (
    ReportAssetBundle,
    benchmark_summary_row,
    export_report_like_figures,
    write_report_snippet,
    write_summary_csv,
    write_summary_markdown,
)
from pkg.reporting.artifacts import _json_default
from pkg.reporting.exports import load_v2_artifact


def main() -> None:
    suite_name = "reward_protocol_ablation_v2"
    spec_names = [
        "reward_protocol_softplus_scaled_v2",
        "reward_protocol_zscore_v2",
        "reward_protocol_rank_v2",
    ]
    suite_bundle = ReportAssetBundle.create(suite_name)
    manifest = {"suite_name": suite_name, "specs": {}}

    for spec_name in spec_names:
        spec_bundle = suite_bundle.for_spec(spec_name)
        rows = []
        spec_exports = {}
        for benchmark in BENCHMARKS:
            _, payload = load_v2_artifact(spec_name, benchmark)
            rows.append(benchmark_summary_row(benchmark, payload["summary"]))
            spec_exports[benchmark] = export_report_like_figures(
                spec_name=spec_name,
                benchmark_name=benchmark,
                suite_name=suite_name,
                bundle=spec_bundle,
            )

        summary_csv = write_summary_csv(spec_bundle.tables_root / "summary_metrics.csv", rows)
        summary_md = write_summary_markdown(spec_bundle.tables_root / "summary_metrics.md", rows)
        snippet_md = write_report_snippet(spec_bundle.snippets_root / "report_snippet.md", spec_name, rows)
        manifest["specs"][spec_name] = {
            "exports": spec_exports,
            "summary_csv": str(summary_csv),
            "summary_md": str(summary_md),
            "report_snippet": str(snippet_md),
        }

    manifest_path = suite_bundle.write_manifest(manifest)
    print(json.dumps({"manifest": str(manifest_path), "specs": list(manifest["specs"].keys())}, indent=2, default=_json_default))


if __name__ == "__main__":
    main()
