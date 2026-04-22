from __future__ import annotations

import csv

import numpy as np

from pkg.bo import run_single_trial
from pkg.experiments.specs import ExperimentSpec
from pkg.reporting import ArtifactRun, plot_diagnostics, plot_regret_comparison, plot_reward_histogram


def _format_bar(current: int, total: int, width: int = 24) -> str:
    if total <= 0:
        return "-" * width
    filled = int(width * current / total)
    return "#" * filled + "-" * (width - filled)


def _summarize_trials(trials: list[dict]) -> dict:
    regrets = np.stack([trial["regrets"] for trial in trials], axis=0)
    best_values = np.stack([trial["best_values"] for trial in trials], axis=0)
    queried_values = np.stack([trial["queried_values"] for trial in trials], axis=0)
    floor_reward_flags = np.stack([trial["floor_reward_flags"] for trial in trials], axis=0)
    step_reward_std = np.stack([trial["step_reward_std"] for trial in trials], axis=0)
    step_improvement_std = np.stack([trial["step_improvement_std"] for trial in trials], axis=0)
    same_mask_repeat_std = np.stack([trial["same_mask_repeat_std"] for trial in trials], axis=0)
    proxy_actual_gain_corr = np.asarray([trial["proxy_actual_gain_corr"] for trial in trials], dtype=np.float32)
    return {
        "mean_regret": regrets.mean(axis=0),
        "std_regret": regrets.std(axis=0),
        "mean_best_value": best_values.mean(axis=0),
        "std_best_value": best_values.std(axis=0),
        "mean_query_value": queried_values.mean(axis=0),
        "std_query_value": queried_values.std(axis=0),
        "final_regret_mean": float(regrets[:, -1].mean()),
        "final_regret_std": float(regrets[:, -1].std()),
        "floor_reward_fraction_mean": float(floor_reward_flags.mean()),
        "step_reward_std_mean": float(step_reward_std.mean()),
        "step_improvement_std_mean": float(step_improvement_std.mean()),
        "same_mask_repeat_std_mean": float(same_mask_repeat_std.mean()),
        "proxy_actual_gain_corr_mean": float(np.nanmean(proxy_actual_gain_corr)),
        "total_time_mean_sec": float(np.mean([trial["total_wall_time_sec"] for trial in trials])),
    }


def _write_trial_csv(artifact_run: ArtifactRun, random_trials: list[dict], gfn_trials: list[dict]) -> None:
    rows = []
    for trial in random_trials + gfn_trials:
        rows.append(
            {
                "benchmark": trial["benchmark"],
                "method": trial["method"],
                "seed": trial["seed"],
                "final_regret": float(trial["regrets"][-1]),
                "final_best_value": float(trial["best_values"][-1]),
                "total_wall_time_sec": float(trial["total_wall_time_sec"]),
                "mean_iteration_time_sec": float(np.mean(trial["iteration_times_sec"])),
                "mean_surrogate_train_time_sec": float(np.mean(trial["surrogate_train_times_sec"])),
                "mean_gfn_train_time_sec": float(np.mean(trial["gfn_train_times_sec"])),
                "mean_proposal_time_sec": float(np.mean(trial["proposal_times_sec"])),
                "floor_reward_fraction": float(np.mean(trial["floor_reward_flags"])),
                "mean_step_reward_std": float(np.mean(trial["step_reward_std"])),
                "mean_step_improvement_std": float(np.mean(trial["step_improvement_std"])),
                "mean_same_mask_repeat_std": float(np.mean(trial["same_mask_repeat_std"])),
                "proxy_actual_gain_corr": float(trial["proxy_actual_gain_corr"]),
            }
        )
    if not rows:
        return
    path = artifact_run.root / "trial_metrics.csv"
    tmp_path = path.with_suffix(".csv.tmp")
    with tmp_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    tmp_path.replace(path)


def _partial_summary(random_trials: list[dict], gfn_trials: list[dict]) -> dict:
    summary: dict[str, dict | None] = {
        "random": _summarize_trials(random_trials) if random_trials else None,
        "gfn": _summarize_trials(gfn_trials) if gfn_trials else None,
    }
    if random_trials and gfn_trials:
        summary["tradeoff"] = {
            "regret_gain_of_gfn": float(summary["random"]["final_regret_mean"] - summary["gfn"]["final_regret_mean"]),
            "slowdown_factor": float(summary["gfn"]["total_time_mean_sec"] / summary["random"]["total_time_mean_sec"]),
        }
    else:
        summary["tradeoff"] = None
    return summary


def _persist_progress(
    artifact_run: ArtifactRun,
    spec: ExperimentSpec,
    random_trials: list[dict],
    gfn_trials: list[dict],
    latest_trial: dict,
) -> None:
    trial_name = f"{latest_trial['method']}_seed_{latest_trial['seed']}.json"
    artifact_run.write_json_to_path(artifact_run.trials_root / trial_name, latest_trial)
    plot_diagnostics(
        latest_trial,
        str(artifact_run.trial_figures_root / f"{latest_trial['method']}_seed_{latest_trial['seed']}_diagnostics.png"),
        f"{spec.name}:{spec.benchmark_name}:{latest_trial['method']}:seed={latest_trial['seed']}",
    )
    _write_trial_csv(artifact_run, random_trials, gfn_trials)
    partial_summary = _partial_summary(random_trials, gfn_trials)
    artifact_run.write_json(
        "partial_summary.json",
        {
            "summary": partial_summary,
            "random_trials": random_trials,
            "gfn_trials": gfn_trials,
        },
    )
    if random_trials and gfn_trials:
        plot_regret_comparison(
            summary_by_method=partial_summary,
            output_path=str(artifact_run.figures_root / "regret_comparison_partial.png"),
            title=f"{spec.benchmark_name} | {spec.name} (partial)",
        )
        plot_reward_histogram(
            random_trials=random_trials,
            gfn_trials=gfn_trials,
            output_path=str(artifact_run.figures_root / "reward_hist_partial.png"),
            title=f"{spec.benchmark_name} | {spec.name} (partial reward hist)",
        )


def run_experiment(spec: ExperimentSpec, verbose: bool = False) -> dict:
    artifact_run = ArtifactRun.create(spec.name, spec.benchmark_name)
    artifact_run.write_json("spec.json", spec.to_dict())
    total_trials = 2 * len(spec.seeds)
    completed_trials = 0
    random_trials = []
    for seed in spec.seeds:
        if verbose:
            bar = _format_bar(completed_trials, total_trials)
            print(
                f"[spec {spec.name}:{spec.benchmark_name}] [{bar}] "
                f"starting random seed={seed}",
                flush=True,
            )
        trial = run_single_trial(
            spec,
            "random",
            seed,
            verbose=verbose,
            progress_label=f"{spec.name}:{spec.benchmark_name}:random:seed={seed}",
        )
        random_trials.append(trial)
        _persist_progress(artifact_run, spec, random_trials, gfn_trials=[], latest_trial=trial)
        completed_trials += 1
        if verbose:
            bar = _format_bar(completed_trials, total_trials)
            print(
                f"[spec {spec.name}:{spec.benchmark_name}] [{bar}] "
                f"finished random seed={seed} final_regret={float(trial['regrets'][-1]):.4f}",
                flush=True,
            )

    gfn_trials = []
    for seed in spec.seeds:
        if verbose:
            bar = _format_bar(completed_trials, total_trials)
            print(
                f"[spec {spec.name}:{spec.benchmark_name}] [{bar}] "
                f"starting gfn seed={seed}",
                flush=True,
            )
        trial = run_single_trial(
            spec,
            "gfn",
            seed,
            verbose=verbose,
            progress_label=f"{spec.name}:{spec.benchmark_name}:gfn:seed={seed}",
        )
        gfn_trials.append(trial)
        _persist_progress(artifact_run, spec, random_trials, gfn_trials, latest_trial=trial)
        completed_trials += 1
        if verbose:
            bar = _format_bar(completed_trials, total_trials)
            print(
                f"[spec {spec.name}:{spec.benchmark_name}] [{bar}] "
                f"finished gfn seed={seed} final_regret={float(trial['regrets'][-1]):.4f}",
                flush=True,
            )

    summary = {
        "random": _summarize_trials(random_trials),
        "gfn": _summarize_trials(gfn_trials),
    }
    summary["tradeoff"] = {
        "regret_gain_of_gfn": float(summary["random"]["final_regret_mean"] - summary["gfn"]["final_regret_mean"]),
        "slowdown_factor": float(summary["gfn"]["total_time_mean_sec"] / summary["random"]["total_time_mean_sec"]),
    }

    artifact_run.write_json(
        "summary.json",
        {
            "summary": summary,
            "random_trials": random_trials,
            "gfn_trials": gfn_trials,
        },
    )
    _write_trial_csv(artifact_run, random_trials, gfn_trials)
    plot_regret_comparison(
        summary_by_method=summary,
        output_path=str(artifact_run.figures_root / "regret_comparison.png"),
        title=f"{spec.benchmark_name} | {spec.name}",
    )
    plot_reward_histogram(
        random_trials=random_trials,
        gfn_trials=gfn_trials,
        output_path=str(artifact_run.figures_root / "reward_hist.png"),
        title=f"{spec.benchmark_name} | {spec.name}",
    )
    if verbose:
        print(
            f"[spec {spec.name}:{spec.benchmark_name}] completed "
            f"regret_gain={summary['tradeoff']['regret_gain_of_gfn']:.4f} "
            f"slowdown={summary['tradeoff']['slowdown_factor']:.2f}x",
            flush=True,
        )
    return {"artifact_dir": str(artifact_run.root), "summary": summary}
