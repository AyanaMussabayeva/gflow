from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pkg.benchmarks import Benchmark
from pkg.masks import MaskSpaceSpec, sample_random_mask_bits
from pkg.rewards.protocols import RewardSpec
from pkg.surrogates import SurrogateBundle


@dataclass
class StepContext:
    candidate_pool: np.ndarray
    heldout_masks: np.ndarray


@dataclass
class MaskEvaluation:
    mask_bits: np.ndarray
    x_next: np.ndarray
    masked_pred: float
    proxy_mean: float
    proxy_std: float
    improvement: float
    reward: float


def sample_uniform(benchmark: Benchmark, n: int, rng: np.random.Generator) -> np.ndarray:
    return benchmark.lower + (benchmark.upper - benchmark.lower) * rng.random((n, benchmark.dim), dtype=np.float32)


def build_step_context(
    benchmark: Benchmark,
    mask_space: MaskSpaceSpec,
    n_candidates: int,
    heldout_mask_samples: int,
    rng_candidates: np.random.Generator,
    rng_masks: np.random.Generator,
) -> StepContext:
    candidate_pool = sample_uniform(benchmark, n_candidates, rng_candidates)
    heldout_masks = np.stack(
        [sample_random_mask_bits(mask_space, rng_masks) for _ in range(heldout_mask_samples)],
        axis=0,
    ).astype(np.float32)
    return StepContext(candidate_pool=candidate_pool, heldout_masks=heldout_masks)


def _improvement_stats(
    bundle: SurrogateBundle,
    step_context: StepContext,
    mask_bits: np.ndarray,
    y_best: float,
    reward_spec: RewardSpec,
) -> tuple[np.ndarray, int, float, float]:
    masked_preds = bundle.predict_masked(step_context.candidate_pool, mask_bits)
    best_idx = int(np.argmax(masked_preds))
    x_best = step_context.candidate_pool[best_idx : best_idx + 1]
    heldout_preds = np.asarray(
        [float(bundle.predict_masked(x_best, heldout_mask)[0]) for heldout_mask in step_context.heldout_masks],
        dtype=np.float32,
    )
    proxy_mean = float(np.mean(heldout_preds))
    proxy_std = float(np.std(heldout_preds))
    improvement = proxy_mean + reward_spec.proxy_beta * proxy_std - float(y_best)
    return masked_preds, best_idx, proxy_mean, proxy_std + 0.0 * improvement


def evaluate_mask_candidates(
    bundle: SurrogateBundle,
    step_context: StepContext,
    mask_bits_list: list[np.ndarray],
    y_best: float,
    reward_spec: RewardSpec,
) -> list[MaskEvaluation]:
    return evaluate_mask_candidates_with_contexts(
        bundle=bundle,
        step_contexts=[step_context] * len(mask_bits_list),
        mask_bits_list=mask_bits_list,
        y_best=y_best,
        reward_spec=reward_spec,
    )


def evaluate_mask_candidates_with_contexts(
    bundle: SurrogateBundle,
    step_contexts: list[StepContext],
    mask_bits_list: list[np.ndarray],
    y_best: float,
    reward_spec: RewardSpec,
) -> list[MaskEvaluation]:
    if len(step_contexts) != len(mask_bits_list):
        raise ValueError("step_contexts and mask_bits_list must have the same length.")

    raw = []
    for step_context, mask_bits in zip(step_contexts, mask_bits_list):
        masked_preds = bundle.predict_masked(step_context.candidate_pool, mask_bits)
        best_idx = int(np.argmax(masked_preds))
        x_best = step_context.candidate_pool[best_idx : best_idx + 1]
        heldout_preds = np.asarray(
            [float(bundle.predict_masked(x_best, heldout_mask)[0]) for heldout_mask in step_context.heldout_masks],
            dtype=np.float32,
        )
        proxy_mean = float(np.mean(heldout_preds))
        proxy_std = float(np.std(heldout_preds))
        improvement = proxy_mean + reward_spec.proxy_beta * proxy_std - float(y_best)
        raw.append(
            {
                "mask_bits": np.asarray(mask_bits, dtype=np.float32),
                "x_next": x_best,
                "masked_pred": float(masked_preds[best_idx]),
                "proxy_mean": proxy_mean,
                "proxy_std": proxy_std,
                "improvement": improvement,
            }
        )
    all_improvements = np.asarray([item["improvement"] for item in raw], dtype=np.float32)
    evaluations = []
    for item in raw:
        reward = reward_spec.reward_from_improvements(float(item["improvement"]), all_improvements)
        evaluations.append(MaskEvaluation(reward=reward, **item))
    return evaluations


def evaluate_mask(
    bundle: SurrogateBundle,
    step_context: StepContext,
    mask_bits: np.ndarray,
    y_best: float,
    reward_spec: RewardSpec,
) -> MaskEvaluation:
    return evaluate_mask_candidates(bundle, step_context, [mask_bits], y_best, reward_spec)[0]
