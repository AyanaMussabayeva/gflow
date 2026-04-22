from __future__ import annotations

import csv
from pathlib import Path


def benchmark_summary_row(benchmark: str, summary: dict) -> dict:
    return {
        "benchmark": benchmark,
        "random_final_regret_mean": float(summary["random"]["final_regret_mean"]),
        "random_final_regret_std": float(summary["random"]["final_regret_std"]),
        "gfn_final_regret_mean": float(summary["gfn"]["final_regret_mean"]),
        "gfn_final_regret_std": float(summary["gfn"]["final_regret_std"]),
        "regret_gain_of_gfn": float(summary["tradeoff"]["regret_gain_of_gfn"]),
        "slowdown_factor": float(summary["tradeoff"]["slowdown_factor"]),
        "random_floor_reward_fraction": float(summary["random"]["floor_reward_fraction_mean"]),
        "gfn_floor_reward_fraction": float(summary["gfn"]["floor_reward_fraction_mean"]),
        "random_improvement_std": float(summary["random"]["step_improvement_std_mean"]),
        "gfn_improvement_std": float(summary["gfn"]["step_improvement_std_mean"]),
        "random_repeat_std": float(summary["random"]["same_mask_repeat_std_mean"]),
        "gfn_repeat_std": float(summary["gfn"]["same_mask_repeat_std_mean"]),
        "random_proxy_actual_gain_corr": float(summary["random"]["proxy_actual_gain_corr_mean"]),
        "gfn_proxy_actual_gain_corr": float(summary["gfn"]["proxy_actual_gain_corr_mean"]),
    }


def write_summary_csv(path: Path, rows: list[dict]) -> Path:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return path


def write_summary_markdown(path: Path, rows: list[dict]) -> Path:
    lines = [
        "| Benchmark | Random final regret | GFN final regret | Regret gain of GFN | Slowdown | Random floor frac | GFN floor frac | Random corr | GFN corr |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {benchmark} | {random_final_regret_mean:.4f} +- {random_final_regret_std:.4f} | "
            "{gfn_final_regret_mean:.4f} +- {gfn_final_regret_std:.4f} | {regret_gain_of_gfn:.4f} | "
            "{slowdown_factor:.2f}x | {random_floor_reward_fraction:.4f} | {gfn_floor_reward_fraction:.4f} | "
            "{random_proxy_actual_gain_corr:.4f} | {gfn_proxy_actual_gain_corr:.4f} |".format(
                **row
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def write_report_snippet(path: Path, spec_name: str, rows: list[dict]) -> Path:
    lines = [f"# {spec_name} report snippet", ""]
    for row in rows:
        lines.append(
            (
                f"{row['benchmark']}: random final regret {row['random_final_regret_mean']:.4f} +- "
                f"{row['random_final_regret_std']:.4f}, GFlowNet final regret {row['gfn_final_regret_mean']:.4f} +- "
                f"{row['gfn_final_regret_std']:.4f}, regret gain of GFN {row['regret_gain_of_gfn']:.4f}, "
                f"slowdown {row['slowdown_factor']:.2f}x, proxy->actual corr "
                f"{row['random_proxy_actual_gain_corr']:.4f} (random) vs {row['gfn_proxy_actual_gain_corr']:.4f} (GFN)."
            )
        )
    lines.append("")
    lines.append(
        "Diagnostic note: floor reward fraction and same-mask repeat std are included in the CSV/MD tables to quantify reward collapse and reward noise."
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
