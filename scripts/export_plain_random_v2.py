from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pkg.reporting.artifacts import _json_default
from pkg.reporting.exports import ReportAssetBundle, load_suite_manifest, load_v2_artifact, select_representative_trial
from pkg.reporting.figures import plot_diagnostics, plot_named_regret_comparison


def _summary_row(benchmark: str, plain_summary: dict, main_summary: dict) -> dict:
    plain = plain_summary["random"]
    matched_random = main_summary["random"]
    gfn = main_summary["gfn"]
    return {
        "benchmark": benchmark,
        "plain_random_final_regret_mean": float(plain["final_regret_mean"]),
        "plain_random_final_regret_std": float(plain["final_regret_std"]),
        "matched_random_final_regret_mean": float(matched_random["final_regret_mean"]),
        "matched_random_final_regret_std": float(matched_random["final_regret_std"]),
        "gfn_final_regret_mean": float(gfn["final_regret_mean"]),
        "gfn_final_regret_std": float(gfn["final_regret_std"]),
        "matched_random_gain_vs_plain": float(plain["final_regret_mean"] - matched_random["final_regret_mean"]),
        "gfn_gain_vs_plain": float(plain["final_regret_mean"] - gfn["final_regret_mean"]),
        "matched_random_slowdown_vs_plain": float(matched_random["total_time_mean_sec"] / plain["total_time_mean_sec"]),
        "gfn_slowdown_vs_plain": float(gfn["total_time_mean_sec"] / plain["total_time_mean_sec"]),
    }


def _write_csv(path: Path, rows: list[dict]) -> Path:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return path


def _write_markdown(path: Path, rows: list[dict]) -> Path:
    lines = [
        "| Benchmark | Plain random | Matched random | GFN | Matched gain vs plain | GFN gain vs plain | Matched slowdown | GFN slowdown |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {benchmark} | {plain_random_final_regret_mean:.4f} +- {plain_random_final_regret_std:.4f} | "
            "{matched_random_final_regret_mean:.4f} +- {matched_random_final_regret_std:.4f} | "
            "{gfn_final_regret_mean:.4f} +- {gfn_final_regret_std:.4f} | "
            "{matched_random_gain_vs_plain:.4f} | {gfn_gain_vs_plain:.4f} | "
            "{matched_random_slowdown_vs_plain:.2f}x | {gfn_slowdown_vs_plain:.2f}x |".format(
                **row
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _write_snippet(path: Path, rows: list[dict]) -> Path:
    lines = ["# plain_random_v2 report snippet", ""]
    for row in rows:
        lines.append(
            (
                f"{row['benchmark']}: plain random final regret "
                f"{row['plain_random_final_regret_mean']:.4f} +- {row['plain_random_final_regret_std']:.4f}, "
                f"matched random {row['matched_random_final_regret_mean']:.4f} +- {row['matched_random_final_regret_std']:.4f}, "
                f"GFN {row['gfn_final_regret_mean']:.4f} +- {row['gfn_final_regret_std']:.4f}. "
                f"Matched random gain vs plain: {row['matched_random_gain_vs_plain']:.4f}; "
                f"GFN gain vs plain: {row['gfn_gain_vs_plain']:.4f}."
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite-name", default="plain_random_v2")
    parser.add_argument("--main-suite-name", default="main_v2")
    parser.add_argument("--suite-manifest", default=None)
    parser.add_argument("--main-suite-manifest", default=None)
    args = parser.parse_args()

    plain_suite_manifest, _ = load_suite_manifest(args.suite_name, suite_manifest_path=args.suite_manifest)
    main_suite_manifest, _ = load_suite_manifest(args.main_suite_name, suite_manifest_path=args.main_suite_manifest)

    bundle = ReportAssetBundle.create(args.suite_name)
    manifest = {
        "suite_name": args.suite_name,
        "suite_manifest": str(plain_suite_manifest),
        "main_suite_name": args.main_suite_name,
        "main_suite_manifest": str(main_suite_manifest),
        "exports": {},
    }

    rows = []
    for benchmark in ["branin", "hartmann6", "ackley10"]:
        plain_root, plain_payload = load_v2_artifact(
            "plain_random_v2",
            benchmark,
            suite_name=args.suite_name,
            suite_manifest_path=str(plain_suite_manifest),
        )
        main_root, main_payload = load_v2_artifact(
            "main_v2",
            benchmark,
            suite_name=args.main_suite_name,
            suite_manifest_path=str(main_suite_manifest),
        )

        comparison_path = bundle.img_root / f"{benchmark}_plain_random_vs_main.png"
        plot_named_regret_comparison(
            method_summaries={
                "plain random (1 mask)": plain_payload["summary"]["random"],
                "matched random (best-of-K)": main_payload["summary"]["random"],
                "GFN": main_payload["summary"]["gfn"],
            },
            output_path=str(comparison_path),
            title=f"{benchmark}: plain random vs matched random vs GFN",
        )

        plain_trial = select_representative_trial(plain_payload["random_trials"])
        matched_random_trial = select_representative_trial(main_payload["random_trials"])
        plain_diag_path = bundle.img_root / f"{benchmark}_plain_random_diagnostics.png"
        matched_diag_path = bundle.img_root / f"{benchmark}_matched_random_diagnostics.png"
        plot_diagnostics(plain_trial, str(plain_diag_path), f"{benchmark}: representative plain-random seed")
        plot_diagnostics(
            matched_random_trial,
            str(matched_diag_path),
            f"{benchmark}: representative matched-random seed",
        )

        rows.append(_summary_row(benchmark, plain_payload["summary"], main_payload["summary"]))
        manifest["exports"][benchmark] = {
            "plain_random_artifact_root": str(plain_root),
            "main_v2_artifact_root": str(main_root),
            "comparison_plot": str(comparison_path),
            "plain_random_diagnostics": str(plain_diag_path),
            "matched_random_diagnostics": str(matched_diag_path),
        }

    manifest["summary_csv"] = str(_write_csv(bundle.tables_root / "plain_random_vs_main.csv", rows))
    manifest["summary_md"] = str(_write_markdown(bundle.tables_root / "plain_random_vs_main.md", rows))
    manifest["report_snippet"] = str(_write_snippet(bundle.snippets_root / "plain_random_vs_main.md", rows))
    manifest_path = bundle.write_manifest(manifest)
    print(json.dumps({"manifest": str(manifest_path), "exports": manifest["exports"]}, indent=2, default=_json_default))


if __name__ == "__main__":
    main()
