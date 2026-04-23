from __future__ import annotations

import numpy as np

try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError:  # pragma: no cover
    plt = None


def plot_regret_comparison(summary_by_method: dict, output_path: str, title: str) -> None:
    if plt is None:
        raise ModuleNotFoundError("matplotlib is required for plotting.")
    x_axis = np.arange(1, len(summary_by_method["random"]["mean_regret"]) + 1)
    plt.figure(figsize=(8, 5))
    for method, color in [("random", "tab:blue"), ("gfn", "tab:orange")]:
        mean = np.asarray(summary_by_method[method]["mean_regret"], dtype=float)
        std = np.asarray(summary_by_method[method]["std_regret"], dtype=float)
        plt.plot(x_axis, mean, label=method, color=color)
        plt.fill_between(x_axis, mean - std, mean + std, alpha=0.2, color=color)
    plt.xlabel("BO iteration")
    plt.ylabel("Simple regret")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_named_regret_comparison(
    method_summaries: dict[str, dict],
    output_path: str,
    title: str,
) -> None:
    if plt is None:
        raise ModuleNotFoundError("matplotlib is required for plotting.")
    if not method_summaries:
        return

    horizon = min(len(summary["mean_regret"]) for summary in method_summaries.values())
    x_axis = np.arange(1, horizon + 1)
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown"]

    plt.figure(figsize=(8, 5))
    for idx, (label, summary) in enumerate(method_summaries.items()):
        mean = np.asarray(summary["mean_regret"], dtype=float)[:horizon]
        std = np.asarray(summary["std_regret"], dtype=float)[:horizon]
        color = colors[idx % len(colors)]
        plt.plot(x_axis, mean, label=label, color=color)
        plt.fill_between(x_axis, mean - std, mean + std, alpha=0.2, color=color)
    plt.xlabel("BO iteration")
    plt.ylabel("Simple regret")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_diagnostics(trial: dict, output_path: str, title: str) -> None:
    if plt is None:
        raise ModuleNotFoundError("matplotlib is required for plotting.")
    steps = np.arange(1, len(trial["proxy_rewards"]) + 1)
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.plot(steps, trial["proxy_rewards"], label="Proxy reward")
    plt.plot(steps, trial["proxy_improvements"], label="Proxy improvement")
    plt.xlabel("BO iteration")
    plt.title(f"{title}: proposal diagnostics")
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(steps, trial["queried_values"], label="Observed y_next")
    plt.plot(steps, trial["best_values"], label="Best-so-far")
    plt.xlabel("BO iteration")
    plt.title(f"{title}: oracle feedback")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_reward_histogram(random_trials: list[dict], gfn_trials: list[dict], output_path: str, title: str) -> None:
    if plt is None:
        raise ModuleNotFoundError("matplotlib is required for plotting.")
    if not random_trials or not gfn_trials:
        return
    random_rewards = np.concatenate([np.asarray(trial["proxy_rewards"], dtype=float) for trial in random_trials], axis=0)
    gfn_rewards = np.concatenate([np.asarray(trial["proxy_rewards"], dtype=float) for trial in gfn_trials], axis=0)
    plt.figure(figsize=(7, 4))
    plt.hist(random_rewards, bins=20, alpha=0.6, label="Random masks")
    plt.hist(gfn_rewards, bins=20, alpha=0.6, label="GFN masks")
    plt.xlabel("Proxy reward")
    plt.ylabel("Count")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_multi_reward_histogram(
    benchmark_payloads: dict[str, dict],
    output_path: str,
    title: str,
) -> None:
    if plt is None:
        raise ModuleNotFoundError("matplotlib is required for plotting.")
    plt.figure(figsize=(7, 4))
    random_rewards = []
    gfn_rewards = []
    for payload in benchmark_payloads.values():
        random_rewards.extend(
            np.concatenate([np.asarray(trial["proxy_rewards"], dtype=float) for trial in payload["random_trials"]]).tolist()
        )
        gfn_rewards.extend(
            np.concatenate([np.asarray(trial["proxy_rewards"], dtype=float) for trial in payload["gfn_trials"]]).tolist()
        )
    plt.hist(np.asarray(random_rewards, dtype=float), bins=24, alpha=0.6, label="Random masks")
    plt.hist(np.asarray(gfn_rewards, dtype=float), bins=24, alpha=0.6, label="GFN masks")
    plt.xlabel("Proxy reward")
    plt.ylabel("Count")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_implementation_comparison(legacy_results: dict, v2_summary: dict, output_path: str, title: str) -> None:
    if plt is None:
        raise ModuleNotFoundError("matplotlib is required for plotting.")
    plt.figure(figsize=(8, 5))
    x_axis = np.arange(1, len(v2_summary["random"]["mean_regret"]) + 1)

    legacy_random = np.mean(
        np.stack([np.asarray(trial["regrets"], dtype=float) for trial in legacy_results["random"]["trials"]], axis=0),
        axis=0,
    )[: len(x_axis)]
    legacy_gfn = np.mean(
        np.stack([np.asarray(trial["regrets"], dtype=float) for trial in legacy_results["gfn"]["trials"]], axis=0),
        axis=0,
    )[: len(x_axis)]
    v2_random = np.asarray(v2_summary["random"]["mean_regret"], dtype=float)
    v2_gfn = np.asarray(v2_summary["gfn"]["mean_regret"], dtype=float)

    plt.plot(x_axis, legacy_random, label="legacy random", color="tab:blue", linestyle="--")
    plt.plot(x_axis, legacy_gfn, label="legacy gfn", color="tab:orange", linestyle="--")
    plt.plot(x_axis, v2_random, label="v2 random", color="tab:blue")
    plt.plot(x_axis, v2_gfn, label="v2 gfn", color="tab:orange")
    plt.xlabel("BO iteration")
    plt.ylabel("Simple regret")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
