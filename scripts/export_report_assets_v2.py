from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pkg.reporting import (
    ReportAssetBundle,
    benchmark_summary_row,
    export_report_like_figures,
    plot_multi_reward_histogram,
    write_report_snippet,
    write_summary_csv,
    write_summary_markdown,
)
from pkg.reporting.exports import load_suite_manifest, load_v2_artifact
from pkg.reporting.artifacts import _json_default


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", default="main_v2")
    parser.add_argument("--suite-name", default="main_v2")
    parser.add_argument("--artifact-dir", default=None)
    parser.add_argument("--suite-manifest", default=None)
    args = parser.parse_args()

    resolved_suite_manifest = None
    if args.artifact_dir is None:
        resolved_suite_manifest, _ = load_suite_manifest(args.suite_name, suite_manifest_path=args.suite_manifest)

    bundle = ReportAssetBundle.create(args.suite_name)
    manifest = {
        "suite_name": args.suite_name,
        "spec_name": args.spec,
        "suite_manifest": str(resolved_suite_manifest) if resolved_suite_manifest is not None else None,
        "exports": {},
    }
    benchmark_payloads = {}
    rows = []
    for benchmark in ["branin", "hartmann6", "ackley10"]:
        artifact_root, payload = load_v2_artifact(
            args.spec,
            benchmark,
            artifact_dir=args.artifact_dir,
            suite_name=args.suite_name,
            suite_manifest_path=str(resolved_suite_manifest) if resolved_suite_manifest is not None else args.suite_manifest,
        )
        benchmark_payloads[benchmark] = payload
        rows.append(benchmark_summary_row(benchmark, payload["summary"]))
        manifest["exports"][benchmark] = export_report_like_figures(
            spec_name=args.spec,
            benchmark_name=benchmark,
            artifact_dir=args.artifact_dir,
            suite_name=args.suite_name,
            suite_manifest_path=str(resolved_suite_manifest) if resolved_suite_manifest is not None else args.suite_manifest,
            bundle=bundle,
        )
        if benchmark == "branin":
            shutil.copyfile(
                manifest["exports"][benchmark]["regret_comparison"],
                bundle.img_root / "branin_regret.png",
            )

    suite_reward_hist = bundle.img_root / "gfn_reward_hist.png"
    plot_multi_reward_histogram(
        benchmark_payloads=benchmark_payloads,
        output_path=str(suite_reward_hist),
        title=f"{args.suite_name}: reward distribution across benchmarks",
    )
    manifest["suite_reward_hist"] = str(suite_reward_hist)
    summary_csv = write_summary_csv(bundle.tables_root / "summary_metrics.csv", rows)
    summary_md = write_summary_markdown(bundle.tables_root / "summary_metrics.md", rows)
    snippet_md = write_report_snippet(bundle.snippets_root / "report_snippet.md", args.spec, rows)
    manifest["summary_csv"] = str(summary_csv)
    manifest["summary_md"] = str(summary_md)
    manifest["report_snippet"] = str(snippet_md)
    manifest_path = bundle.write_manifest(manifest)
    print(json.dumps({"manifest": str(manifest_path), "exports": manifest["exports"]}, indent=2, default=_json_default))


if __name__ == "__main__":
    main()
