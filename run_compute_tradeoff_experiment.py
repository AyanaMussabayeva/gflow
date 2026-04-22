from __future__ import annotations

import argparse
import csv
import json
from time import perf_counter
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np

from gfn_bo_experiments import default_config, plt, run_single_trial


ROOT = Path(__file__).resolve().parent


def profile_config(benchmark_name: str, profile: str, seeds: List[int]):
    cfg = default_config(benchmark_name, seeds=seeds)
    if profile == "full":
        return cfg
    if profile == "smoke":
        cfg.n_iter = min(cfg.n_iter, 4)
        cfg.surrogate_epochs = 40
        cfg.n_candidates = min(cfg.n_candidates, 256)
        cfg.heldout_mask_samples = min(cfg.heldout_mask_samples, 6)
        cfg.random_mask_samples = min(cfg.random_mask_samples, 12)
        cfg.gfn_mask_samples = min(cfg.gfn_mask_samples, 12)
        cfg.gfn_steps = min(cfg.gfn_steps, 8)
        cfg.gfn_batch_size = min(cfg.gfn_batch_size, 6)
        return cfg
    raise ValueError(f"Unsupported profile: {profile}")


def mean_std(values: Iterable[float]) -> Dict[str, float]:
    arr = np.asarray(list(values), dtype=np.float64)
    return {"mean": float(arr.mean()), "std": float(arr.std())}


def trial_record(trial: Dict) -> Dict[str, float | int | str]:
    return {
        "benchmark": str(trial["benchmark"]),
        "method": str(trial["method"]),
        "seed": int(trial["seed"]),
        "final_regret": float(trial["regrets"][-1]),
        "final_best_value": float(trial["best_values"][-1]),
        "total_wall_time_sec": float(trial["total_wall_time_sec"]),
        "mean_iteration_time_sec": float(np.mean(trial["iteration_times_sec"])),
        "mean_surrogate_train_time_sec": float(np.mean(trial["surrogate_train_times_sec"])),
        "mean_gfn_train_time_sec": float(np.mean(trial["gfn_train_times_sec"])),
        "mean_proposal_time_sec": float(np.mean(trial["proposal_times_sec"])),
        "mean_oracle_eval_time_sec": float(np.mean(trial["oracle_eval_times_sec"])),
    }


def summarize_method(trials: List[Dict]) -> Dict[str, float]:
    final_regrets = [float(trial["regrets"][-1]) for trial in trials]
    final_best_values = [float(trial["best_values"][-1]) for trial in trials]
    total_times = [float(trial["total_wall_time_sec"]) for trial in trials]
    iter_times = [float(np.mean(trial["iteration_times_sec"])) for trial in trials]
    surrogate_times = [float(np.mean(trial["surrogate_train_times_sec"])) for trial in trials]
    gfn_train_times = [float(np.mean(trial["gfn_train_times_sec"])) for trial in trials]
    proposal_times = [float(np.mean(trial["proposal_times_sec"])) for trial in trials]

    return {
        "final_regret_mean": mean_std(final_regrets)["mean"],
        "final_regret_std": mean_std(final_regrets)["std"],
        "final_best_mean": mean_std(final_best_values)["mean"],
        "final_best_std": mean_std(final_best_values)["std"],
        "total_time_mean_sec": mean_std(total_times)["mean"],
        "total_time_std_sec": mean_std(total_times)["std"],
        "iter_time_mean_sec": mean_std(iter_times)["mean"],
        "surrogate_time_mean_sec": mean_std(surrogate_times)["mean"],
        "gfn_train_time_mean_sec": mean_std(gfn_train_times)["mean"],
        "proposal_time_mean_sec": mean_std(proposal_times)["mean"],
    }


def run_experiment(benchmarks: List[str], profile: str, seeds: List[int]) -> Dict[str, Dict]:
    results: Dict[str, Dict] = {}
    for benchmark_name in benchmarks:
        print(f"\n=== Benchmark: {benchmark_name} | profile={profile} ===", flush=True)
        cfg = profile_config(benchmark_name, profile=profile, seeds=seeds)
        print(
            "Config: "
            f"n_init={cfg.n_init}, n_iter={cfg.n_iter}, hidden_dim={cfg.surrogate_hidden_dim}, "
            f"candidates={cfg.n_candidates}, gfn_steps={cfg.gfn_steps}, seeds={cfg.seeds}",
            flush=True,
        )

        random_trials = []
        for seed in cfg.seeds:
            label = f"{benchmark_name}:random:seed={seed}"
            print(f"Starting {label}", flush=True)
            start = perf_counter()
            trial = run_single_trial(
                benchmark_name,
                "random",
                seed=seed,
                cfg=cfg,
                collect_timing=True,
                verbose=True,
                progress_label=label,
            )
            random_trials.append(trial)
            print(
                f"Finished {label} in {perf_counter() - start:.1f}s | "
                f"final_regret={float(trial['regrets'][-1]):.4f}",
                flush=True,
            )

        gfn_trials = []
        for seed in cfg.seeds:
            label = f"{benchmark_name}:gfn:seed={seed}"
            print(f"Starting {label}", flush=True)
            start = perf_counter()
            trial = run_single_trial(
                benchmark_name,
                "gfn",
                seed=seed,
                cfg=cfg,
                collect_timing=True,
                verbose=True,
                progress_label=label,
            )
            gfn_trials.append(trial)
            print(
                f"Finished {label} in {perf_counter() - start:.1f}s | "
                f"final_regret={float(trial['regrets'][-1]):.4f}",
                flush=True,
            )
        random_summary = summarize_method(random_trials)
        gfn_summary = summarize_method(gfn_trials)
        extra_time = gfn_summary["total_time_mean_sec"] - random_summary["total_time_mean_sec"]
        regret_gain = random_summary["final_regret_mean"] - gfn_summary["final_regret_mean"]
        print(
            f"Summary for {benchmark_name}: "
            f"random_regret={random_summary['final_regret_mean']:.4f}, "
            f"gfn_regret={gfn_summary['final_regret_mean']:.4f}, "
            f"slowdown={gfn_summary['total_time_mean_sec'] / random_summary['total_time_mean_sec']:.2f}x",
            flush=True,
        )
        results[benchmark_name] = {
            "config": {
                "n_init": cfg.n_init,
                "n_iter": cfg.n_iter,
                "seeds": cfg.seeds,
                "surrogate_hidden_dim": cfg.surrogate_hidden_dim,
                "surrogate_epochs": cfg.surrogate_epochs,
                "n_candidates": cfg.n_candidates,
                "heldout_mask_samples": cfg.heldout_mask_samples,
                "random_mask_samples": cfg.random_mask_samples,
                "gfn_mask_samples": cfg.gfn_mask_samples,
                "gfn_steps": cfg.gfn_steps,
                "gfn_batch_size": cfg.gfn_batch_size,
            },
            "random": {"trials": random_trials, "summary": random_summary},
            "gfn": {"trials": gfn_trials, "summary": gfn_summary},
            "tradeoff": {
                "regret_gain_of_gfn": float(regret_gain),
                "extra_time_sec_for_gfn": float(extra_time),
                "slowdown_factor": float(gfn_summary["total_time_mean_sec"] / random_summary["total_time_mean_sec"]),
                "regret_gain_per_extra_hour": float(regret_gain / extra_time * 3600.0) if extra_time > 0 else 0.0,
            },
        }
    return results


def json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.integer):
        return int(value)
    raise TypeError(f"Unsupported type: {type(value)!r}")


def save_trial_csv(results: Dict[str, Dict], output_dir: Path) -> None:
    rows = []
    for benchmark_name, benchmark_results in results.items():
        for method in ["random", "gfn"]:
            for trial in benchmark_results[method]["trials"]:
                rows.append(trial_record(trial))

    fieldnames = list(rows[0].keys())
    with (output_dir / "trial_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_summary_csv(results: Dict[str, Dict], output_dir: Path) -> None:
    rows = []
    for benchmark_name, benchmark_results in results.items():
        for method in ["random", "gfn"]:
            summary = benchmark_results[method]["summary"]
            rows.append(
                {
                    "benchmark": benchmark_name,
                    "method": method,
                    **summary,
                }
            )
    fieldnames = list(rows[0].keys())
    with (output_dir / "summary_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_tradeoff_csv(results: Dict[str, Dict], output_dir: Path) -> None:
    rows = []
    for benchmark_name, benchmark_results in results.items():
        rows.append({"benchmark": benchmark_name, **benchmark_results["tradeoff"]})
    fieldnames = list(rows[0].keys())
    with (output_dir / "tradeoff_summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_plots(results: Dict[str, Dict], output_dir: Path) -> None:
    if plt is None:
        raise ModuleNotFoundError("matplotlib is required to save plots.")

    benchmarks = list(results.keys())
    x = np.arange(len(benchmarks))
    width = 0.36

    random_times = [results[b]["random"]["summary"]["total_time_mean_sec"] / 60.0 for b in benchmarks]
    gfn_times = [results[b]["gfn"]["summary"]["total_time_mean_sec"] / 60.0 for b in benchmarks]
    random_regrets = [results[b]["random"]["summary"]["final_regret_mean"] for b in benchmarks]
    gfn_regrets = [results[b]["gfn"]["summary"]["final_regret_mean"] for b in benchmarks]

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.bar(x - width / 2, random_times, width=width, label="Random", color="tab:blue")
    plt.bar(x + width / 2, gfn_times, width=width, label="GFlowNet", color="tab:orange")
    plt.xticks(x, benchmarks)
    plt.ylabel("Mean wall-clock time per trial (minutes)")
    plt.title("Compute cost by benchmark")
    plt.legend()
    plt.grid(True, axis="y", alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(x, random_regrets, marker="o", label="Random", color="tab:blue")
    plt.plot(x, gfn_regrets, marker="o", label="GFlowNet", color="tab:orange")
    plt.xticks(x, benchmarks)
    plt.ylabel("Mean final regret")
    plt.title("Final regret by benchmark")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "compute_tradeoff_overview.png", dpi=180, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(9, 5))
    random_iter = [results[b]["random"]["summary"]["iter_time_mean_sec"] for b in benchmarks]
    gfn_iter = [results[b]["gfn"]["summary"]["iter_time_mean_sec"] for b in benchmarks]
    random_surrogate = [results[b]["random"]["summary"]["surrogate_time_mean_sec"] for b in benchmarks]
    gfn_surrogate = [results[b]["gfn"]["summary"]["surrogate_time_mean_sec"] for b in benchmarks]
    gfn_train = [results[b]["gfn"]["summary"]["gfn_train_time_mean_sec"] for b in benchmarks]
    random_proposal = [results[b]["random"]["summary"]["proposal_time_mean_sec"] for b in benchmarks]
    gfn_proposal = [results[b]["gfn"]["summary"]["proposal_time_mean_sec"] for b in benchmarks]

    x2 = np.arange(len(benchmarks))
    plt.bar(x2 - width / 2, random_surrogate, width=width, label="Random surrogate train", color="#4C78A8")
    plt.bar(
        x2 - width / 2,
        random_proposal,
        width=width,
        bottom=random_surrogate,
        label="Random proposal selection",
        color="#9ECAE9",
    )
    plt.bar(x2 + width / 2, gfn_surrogate, width=width, label="GFN surrogate train", color="#F58518")
    plt.bar(
        x2 + width / 2,
        gfn_train,
        width=width,
        bottom=gfn_surrogate,
        label="GFN training",
        color="#FFBF79",
    )
    gfn_bottom = np.asarray(gfn_surrogate) + np.asarray(gfn_train)
    plt.bar(
        x2 + width / 2,
        gfn_proposal,
        width=width,
        bottom=gfn_bottom,
        label="GFN proposal selection",
        color="#FFD7A8",
    )
    plt.xticks(x2, benchmarks)
    plt.ylabel("Mean per-iteration wall-clock time (seconds)")
    plt.title("Where the extra compute goes")
    plt.legend(fontsize=8)
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "compute_tradeoff_breakdown.png", dpi=180, bbox_inches="tight")
    plt.close()


def save_report_snippet(results: Dict[str, Dict], output_dir: Path, profile: str) -> None:
    lines = [
        "# Compute Trade-off Results",
        "",
        f"Profile: `{profile}`",
        "",
        "| Benchmark | Random final regret | GFlowNet final regret | Random time / trial | GFlowNet time / trial | Slowdown | Regret gain of GFlowNet |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for benchmark_name, benchmark_results in results.items():
        random_summary = benchmark_results["random"]["summary"]
        gfn_summary = benchmark_results["gfn"]["summary"]
        tradeoff = benchmark_results["tradeoff"]
        lines.append(
            "| "
            + " | ".join(
                [
                    benchmark_name,
                    f"{random_summary['final_regret_mean']:.4f} +- {random_summary['final_regret_std']:.4f}",
                    f"{gfn_summary['final_regret_mean']:.4f} +- {gfn_summary['final_regret_std']:.4f}",
                    f"{random_summary['total_time_mean_sec'] / 60.0:.2f} min",
                    f"{gfn_summary['total_time_mean_sec'] / 60.0:.2f} min",
                    f"{tradeoff['slowdown_factor']:.2f}x",
                    f"{tradeoff['regret_gain_of_gfn']:.4f}",
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "Positive `regret gain of GFlowNet` means the learned policy achieved lower regret than the random baseline.",
            "",
            "Recommended report interpretation:",
            "- Report both final regret and wall-clock slowdown.",
            "- Treat Ackley-like positive gains as meaningful only if they justify the extra compute.",
            "- For negative-gain benchmarks, the GFlowNet overhead is not justified under the current reward design.",
        ]
    )
    (output_dir / "report_snippet.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def save_readme(output_dir: Path, profile: str, benchmarks: List[str], seeds: List[int]) -> None:
    text = f"""# Compute Trade-off Experiment

This directory stores the outputs of the compute trade-off experiment comparing random mask sampling and the contextual GFlowNet policy.

Configuration:
- profile: `{profile}`
- benchmarks: `{", ".join(benchmarks)}`
- seeds: `{", ".join(map(str, seeds))}`

Files:
- `raw_results.json`: raw trial outputs with regrets and timing arrays
- `trial_metrics.csv`: one row per seed/method/benchmark
- `summary_metrics.csv`: aggregated metrics per method and benchmark
- `tradeoff_summary.csv`: direct random-vs-GFlowNet trade-off metrics
- `compute_tradeoff_overview.png`: runtime and regret overview figure
- `compute_tradeoff_breakdown.png`: per-iteration runtime breakdown figure
- `report_snippet.md`: ready-to-paste report summary
"""
    (output_dir / "README.md").write_text(text, encoding="utf-8")


def print_console_summary(results: Dict[str, Dict]) -> None:
    print("Benchmark    Method   Final regret (mean +- std)    Time/trial    Iter time")
    for benchmark_name, benchmark_results in results.items():
        for method in ["random", "gfn"]:
            summary = benchmark_results[method]["summary"]
            print(
                f"{benchmark_name:<12}{method:<8}"
                f"{summary['final_regret_mean']:.4f} +- {summary['final_regret_std']:.4f}    "
                f"{summary['total_time_mean_sec'] / 60.0:.2f} min    "
                f"{summary['iter_time_mean_sec']:.2f} s"
            )
        tradeoff = benchmark_results["tradeoff"]
        print(
            f"{'':<12}{'delta':<8}"
            f"regret_gain_of_gfn={tradeoff['regret_gain_of_gfn']:.4f}    "
            f"slowdown={tradeoff['slowdown_factor']:.2f}x"
        )


def parse_args():
    parser = argparse.ArgumentParser(description="Run the compute trade-off experiment and save report-ready outputs.")
    parser.add_argument(
        "--profile",
        choices=["smoke", "full"],
        default="full",
        help="Experiment profile. Use 'smoke' for a fast verification run and 'full' for report-ready numbers.",
    )
    parser.add_argument(
        "--benchmarks",
        nargs="+",
        default=["branin", "hartmann6", "ackley10"],
        help="Benchmarks to run.",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[0, 1, 2, 3, 4],
        help="Seeds to run.",
    )
    parser.add_argument(
        "--output-dir",
        default="results/compute_tradeoff/full",
        help="Directory where all outputs will be stored.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    results = run_experiment(benchmarks=args.benchmarks, profile=args.profile, seeds=args.seeds)
    (output_dir / "raw_results.json").write_text(
        json.dumps(results, indent=2, default=json_default) + "\n",
        encoding="utf-8",
    )
    save_trial_csv(results, output_dir)
    save_summary_csv(results, output_dir)
    save_tradeoff_csv(results, output_dir)
    save_plots(results, output_dir)
    save_report_snippet(results, output_dir, args.profile)
    save_readme(output_dir, args.profile, args.benchmarks, args.seeds)
    print_console_summary(results)
    print(f"\nSaved outputs to: {output_dir}")


if __name__ == "__main__":
    main()
