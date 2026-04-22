from __future__ import annotations

from dataclasses import replace

from pkg.experiments.config_loader import load_suite_config
from pkg.experiments.specs import ExperimentSpec, make_experiment_spec
from pkg.rewards import RewardSpec


BENCHMARKS = ["branin", "hartmann6", "ackley10"]


def _with_name(spec: ExperimentSpec, name: str, reward_protocol: str | None = None) -> ExperimentSpec:
    reward_spec = spec.reward_spec if reward_protocol is None else RewardSpec(
        protocol=reward_protocol,
        proxy_beta=spec.reward_spec.proxy_beta,
        reward_temperature=spec.reward_spec.reward_temperature,
        epsilon=spec.reward_spec.epsilon,
    )
    return replace(spec, name=name, reward_spec=reward_spec)


def build_suite_specs(suite_name: str, profile: str) -> list[ExperimentSpec]:
    suite_cfg = load_suite_config(suite_name)
    specs: list[ExperimentSpec] = []
    for entry in suite_cfg["entries"]:
        base = make_experiment_spec(
            benchmark_name=entry["benchmark"],
            name=entry["name"],
            profile=profile,
            reward_protocol=entry.get("reward_protocol", "rank"),
        )
        overrides = dict(entry.get("overrides", {}))
        reward_protocol = overrides.pop("reward_protocol", None)
        spec = _with_name(base, name=entry["name"], reward_protocol=reward_protocol)
        if overrides:
            spec = replace(spec, **overrides)
        specs.append(spec)
    return specs
