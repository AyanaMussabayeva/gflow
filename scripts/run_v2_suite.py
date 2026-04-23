from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pkg import ARTIFACTS_V2_ROOT, build_suite_specs, run_experiment
from pkg.reporting.artifacts import _json_default


def _format_bar(current: int, total: int, width: int = 24) -> str:
    if total <= 0:
        return "-" * width
    filled = int(width * current / total)
    return "#" * filled + "-" * (width - filled)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--suite",
        default="main_v2",
        choices=[
            "main_v2",
            "plain_random_v2",
            "reward_protocol_ablation_v2",
            "pool_ablation_v2",
            "finetune_ablation_v2",
        ],
    )
    parser.add_argument("--profile", default="smoke", choices=["smoke", "full"])
    parser.add_argument("--methods", default="random,gfn")
    args = parser.parse_args()

    specs = build_suite_specs(args.suite, args.profile)
    methods = [item.strip() for item in args.methods.split(",") if item.strip()]
    manifest = {
        "suite": args.suite,
        "profile": args.profile,
        "methods": methods,
        "runs": [],
    }
    suite_dir = ARTIFACTS_V2_ROOT / "_suites" / args.suite / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    suite_dir.mkdir(parents=True, exist_ok=False)
    manifest_path = suite_dir / "manifest.json"

    def write_manifest() -> None:
        tmp_path = manifest_path.with_suffix(".json.tmp")
        tmp_path.write_text(json.dumps(manifest, indent=2, default=_json_default) + "\n", encoding="utf-8")
        tmp_path.replace(manifest_path)

    write_manifest()
    total_specs = len(specs)
    for idx, spec in enumerate(specs, start=1):
        bar = _format_bar(idx - 1, total_specs)
        print(
            f"[suite {args.suite}] [{bar}] starting {idx}/{total_specs} "
            f"{spec.name}:{spec.benchmark_name}",
            flush=True,
        )
        result = run_experiment(spec, verbose=True, methods=methods)
        manifest["runs"].append(
            {
                "spec_name": spec.name,
                "benchmark": spec.benchmark_name,
                "methods": methods,
                "artifact_dir": result["artifact_dir"],
                "summary": result["summary"],
            }
        )
        write_manifest()
        bar = _format_bar(idx, total_specs)
        print(
            f"[suite {args.suite}] [{bar}] finished {idx}/{total_specs} "
            f"{spec.name}:{spec.benchmark_name}",
            flush=True,
        )
    print(json.dumps({"manifest": str(manifest_path), "n_runs": len(manifest["runs"])}, indent=2))


if __name__ == "__main__":
    main()
