from __future__ import annotations

from dataclasses import asdict, dataclass

from pkg.experiments.config_loader import load_benchmark_config, load_profile_config
from pkg.masks import MaskSpaceSpec
from pkg.rewards import RewardSpec


@dataclass(frozen=True)
class ExperimentSpec:
    name: str
    benchmark_name: str
    seeds: list[int]
    n_init: int
    n_iter: int
    mask_space: MaskSpaceSpec
    surrogate_dropout_p: float
    surrogate_epochs: int
    surrogate_lr: float
    n_candidates: int
    heldout_mask_samples: int
    random_mask_samples: int
    gfn_mask_samples: int
    gfn_hidden_size: int
    gfn_steps: int
    gfn_batch_size: int
    gfn_lr: float
    continual_finetune: bool
    shared_step_context: bool
    use_block_mask_training: bool
    repeat_eval_repeats: int
    reward_spec: RewardSpec

    def to_dict(self) -> dict:
        data = asdict(self)
        data["mask_space"] = asdict(self.mask_space)
        data["reward_spec"] = asdict(self.reward_spec)
        return data


def make_experiment_spec(
    benchmark_name: str,
    name: str,
    profile: str = "smoke",
    seeds: list[int] | None = None,
    reward_protocol: str = "rank",
) -> ExperimentSpec:
    benchmark_cfg = load_benchmark_config(benchmark_name)
    profile_cfg = load_profile_config(profile)

    base_seeds = benchmark_cfg["seeds"]
    resolved_seeds = list(base_seeds if seeds is None else seeds)
    scale = profile_cfg.get("scale", {})
    limits = profile_cfg.get("limits", {})

    def scaled_int(key: str, default: int) -> int:
        value = int(benchmark_cfg.get(key, default))
        if key in scale:
            value = max(1, int(round(value * float(scale[key]))))
        if key in limits:
            value = min(value, int(limits[key]))
        return value

    return ExperimentSpec(
        name=name,
        benchmark_name=benchmark_name,
        seeds=resolved_seeds,
        n_init=int(benchmark_cfg["n_init"]),
        n_iter=scaled_int("n_iter", int(benchmark_cfg["n_iter"])),
        mask_space=MaskSpaceSpec(
            hidden_dim=int(benchmark_cfg["mask_space"]["hidden_dim"]),
            block_size=int(benchmark_cfg["mask_space"]["block_size"]),
            keep_prob=float(benchmark_cfg["mask_space"]["keep_prob"]),
        ),
        surrogate_dropout_p=float(benchmark_cfg["surrogate_dropout_p"]),
        surrogate_epochs=scaled_int("surrogate_epochs", int(benchmark_cfg["surrogate_epochs"])),
        surrogate_lr=float(benchmark_cfg["surrogate_lr"]),
        n_candidates=scaled_int("n_candidates", int(benchmark_cfg["n_candidates"])),
        heldout_mask_samples=scaled_int("heldout_mask_samples", int(benchmark_cfg["heldout_mask_samples"])),
        random_mask_samples=scaled_int("random_mask_samples", int(benchmark_cfg["random_mask_samples"])),
        gfn_mask_samples=scaled_int("gfn_mask_samples", int(benchmark_cfg["gfn_mask_samples"])),
        gfn_hidden_size=int(benchmark_cfg["gfn_hidden_size"]),
        gfn_steps=scaled_int("gfn_steps", int(benchmark_cfg["gfn_steps"])),
        gfn_batch_size=scaled_int("gfn_batch_size", int(benchmark_cfg["gfn_batch_size"])),
        gfn_lr=float(benchmark_cfg["gfn_lr"]),
        continual_finetune=bool(benchmark_cfg["continual_finetune"]),
        shared_step_context=bool(benchmark_cfg["shared_step_context"]),
        use_block_mask_training=bool(benchmark_cfg["use_block_mask_training"]),
        repeat_eval_repeats=scaled_int("repeat_eval_repeats", int(benchmark_cfg["repeat_eval_repeats"])),
        reward_spec=RewardSpec(
            protocol=reward_protocol,
            proxy_beta=float(benchmark_cfg["reward_spec"]["proxy_beta"]),
            reward_temperature=float(benchmark_cfg["reward_spec"]["reward_temperature"]),
            epsilon=float(benchmark_cfg["reward_spec"]["epsilon"]),
        ),
    )
