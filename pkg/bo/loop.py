from __future__ import annotations

from time import perf_counter

import numpy as np

from pkg.benchmarks import get_benchmark
from pkg.bo.step_context import build_step_context, dataset_context
from pkg.experiments.specs import ExperimentSpec
from pkg.policies import sample_random_policy_masks, train_contextual_gflownet
from pkg.rewards import evaluate_mask_candidates, evaluate_mask_candidates_with_contexts
from pkg.rng import SeedStreams, set_global_seed
from pkg.surrogates import train_surrogate


def _select_best(evaluations):
    return max(evaluations, key=lambda evaluation: float(evaluation.reward))


def _corrcoef_or_nan(x: list[float], y: list[float]) -> float:
    x_arr = np.asarray(x, dtype=np.float32)
    y_arr = np.asarray(y, dtype=np.float32)
    if x_arr.size < 2 or float(np.std(x_arr)) < 1e-8 or float(np.std(y_arr)) < 1e-8:
        return float("nan")
    return float(np.corrcoef(x_arr, y_arr)[0, 1])


def _format_bar(current: int, total: int, width: int = 24) -> str:
    if total <= 0:
        return "-" * width
    filled = int(width * current / total)
    return "#" * filled + "-" * (width - filled)


def run_single_trial(
    spec: ExperimentSpec,
    method: str,
    seed: int,
    verbose: bool = False,
    progress_label: str | None = None,
) -> dict:
    benchmark = get_benchmark(spec.benchmark_name)
    streams = SeedStreams(seed)
    set_global_seed(seed)
    label = progress_label or f"{spec.name}:{spec.benchmark_name}:{method}:seed={seed}"

    rng_init = streams.generator("init_design")
    x = benchmark.lower + (benchmark.upper - benchmark.lower) * rng_init.random(
        (spec.n_init, benchmark.dim), dtype=np.float32
    )
    y = benchmark.evaluate(x)

    gfn = None
    regrets = []
    best_values = []
    queried_values = []
    actual_gains = []
    proxy_rewards = []
    proxy_improvements = []
    floor_reward_flags = []
    step_reward_std = []
    step_improvement_std = []
    same_mask_repeat_std = []
    surrogate_train_times = []
    gfn_train_times = []
    proposal_times = []
    iteration_times = []

    trial_start = perf_counter()
    for iter_idx in range(spec.n_iter):
        iter_start = perf_counter()
        surrogate_start = perf_counter()
        bundle = train_surrogate(
            benchmark=benchmark,
            x=x,
            y=y,
            mask_space=spec.mask_space,
            dropout_p=spec.surrogate_dropout_p,
            epochs=spec.surrogate_epochs,
            lr=spec.surrogate_lr,
            rng=streams.generator(f"surrogate_train_{iter_idx}"),
            use_block_mask_training=spec.use_block_mask_training,
        )
        surrogate_train_times.append(perf_counter() - surrogate_start)

        context = dataset_context(
            benchmark=benchmark,
            x=x,
            y=y,
            standardizer=bundle.standardizer,
            n_init=spec.n_init,
            n_iter=spec.n_iter,
        )
        shared_step_context = build_step_context(
            benchmark=benchmark,
            mask_space=spec.mask_space,
            n_candidates=spec.n_candidates,
            heldout_mask_samples=spec.heldout_mask_samples,
            rng_candidates=streams.generator(f"candidate_pool_{iter_idx}"),
            rng_masks=streams.generator(f"heldout_masks_{iter_idx}"),
        )
        y_best = float(np.max(y))

        def build_mask_contexts(count: int, substep_idx: int, stage: str) -> list:
            if spec.shared_step_context:
                return [shared_step_context] * count
            contexts = []
            for mask_idx in range(count):
                contexts.append(
                    build_step_context(
                        benchmark=benchmark,
                        mask_space=spec.mask_space,
                        n_candidates=spec.n_candidates,
                        heldout_mask_samples=spec.heldout_mask_samples,
                        rng_candidates=streams.generator(
                            f"{stage}_candidate_pool_{iter_idx}_{substep_idx}_{mask_idx}"
                        ),
                        rng_masks=streams.generator(
                            f"{stage}_heldout_masks_{iter_idx}_{substep_idx}_{mask_idx}"
                        ),
                    )
                )
            return contexts

        proposal_start = perf_counter()
        if method == "random":
            mask_bits_list = sample_random_policy_masks(
                spec.mask_space,
                spec.random_mask_samples,
                streams.generator(f"random_policy_{iter_idx}"),
            )
            evaluations = evaluate_mask_candidates_with_contexts(
                bundle=bundle,
                step_contexts=build_mask_contexts(len(mask_bits_list), 0, "random_eval"),
                mask_bits_list=mask_bits_list,
                y_best=y_best,
                reward_spec=spec.reward_spec,
            )
        elif method == "gfn":
            if not spec.continual_finetune:
                gfn = None
            gfn_start = perf_counter()
            gfn, train_stats = train_contextual_gflownet(
                bundle=bundle,
                context=context,
                y_best=y_best,
                reward_spec=spec.reward_spec,
                mask_samples_per_step=spec.gfn_mask_samples,
                n_steps=spec.gfn_steps,
                batch_size=spec.gfn_batch_size,
                hidden_size=spec.gfn_hidden_size,
                lr=spec.gfn_lr,
                rng=streams.generator(f"gfn_train_{iter_idx}"),
                step_context_builder=lambda count, step_idx: build_mask_contexts(count, step_idx, "gfn_train"),
                gfn=gfn,
            )
            gfn_train_times.append(perf_counter() - gfn_start)
            mask_bits_list = []
            for sample_idx in range(spec.gfn_mask_samples):
                mask_bits, _ = gfn.sample_trajectory(
                    context=context,
                    mask_space=spec.mask_space,
                    rng=streams.generator(f"gfn_policy_{iter_idx}_{sample_idx}"),
                )
                mask_bits_list.append(mask_bits)
            evaluations = evaluate_mask_candidates_with_contexts(
                bundle=bundle,
                step_contexts=build_mask_contexts(len(mask_bits_list), 0, "gfn_eval"),
                mask_bits_list=mask_bits_list,
                y_best=y_best,
                reward_spec=spec.reward_spec,
            )
            train_stats
        else:
            raise ValueError("method must be 'random' or 'gfn'")
        proposal_times.append(perf_counter() - proposal_start)
        if method == "random":
            gfn_train_times.append(0.0)

        best_eval = _select_best(evaluations)
        repeat_improvements = []
        for repeat_idx in range(spec.repeat_eval_repeats):
            repeat_context = build_step_context(
                benchmark=benchmark,
                mask_space=spec.mask_space,
                n_candidates=spec.n_candidates,
                heldout_mask_samples=spec.heldout_mask_samples,
                rng_candidates=streams.generator(f"repeat_candidate_pool_{iter_idx}_{repeat_idx}"),
                rng_masks=streams.generator(f"repeat_heldout_masks_{iter_idx}_{repeat_idx}"),
            )
            repeat_eval = evaluate_mask_candidates(
                bundle=bundle,
                step_context=repeat_context,
                mask_bits_list=[best_eval.mask_bits],
                y_best=y_best,
                reward_spec=spec.reward_spec,
            )[0]
            repeat_improvements.append(float(repeat_eval.improvement))
        same_mask_repeat_std.append(float(np.std(repeat_improvements)))
        step_reward_std.append(float(np.std([item.reward for item in evaluations])))
        step_improvement_std.append(float(np.std([item.improvement for item in evaluations])))
        floor_reward_flags.append(
            float(np.mean([item.reward <= spec.reward_spec.epsilon * 1.01 for item in evaluations]))
        )

        x_next = np.asarray(best_eval.x_next, dtype=np.float32)
        y_next = benchmark.evaluate(x_next)
        x = np.concatenate([x, x_next], axis=0)
        y = np.concatenate([y, y_next], axis=0)

        best_so_far = float(np.max(y))
        actual_gain = float(y_next[0] - y_best)
        regrets.append(float(benchmark.true_max - best_so_far))
        best_values.append(best_so_far)
        queried_values.append(float(y_next[0]))
        actual_gains.append(actual_gain)
        proxy_rewards.append(float(best_eval.reward))
        proxy_improvements.append(float(best_eval.improvement))
        iteration_times.append(perf_counter() - iter_start)

        if verbose:
            elapsed = perf_counter() - trial_start
            bar = _format_bar(iter_idx + 1, spec.n_iter)
            print(
                f"[{label}] [{bar}] {iter_idx + 1}/{spec.n_iter} "
                f"elapsed={elapsed:.1f}s best={best_so_far:.4f} regret={regrets[-1]:.4f}",
                flush=True,
            )

    proxy_actual_gain_corr = _corrcoef_or_nan(proxy_improvements, actual_gains)

    return {
        "seed": seed,
        "method": method,
        "benchmark": spec.benchmark_name,
        "spec_name": spec.name,
        "regrets": np.asarray(regrets, dtype=np.float32),
        "best_values": np.asarray(best_values, dtype=np.float32),
        "queried_values": np.asarray(queried_values, dtype=np.float32),
        "actual_gains": np.asarray(actual_gains, dtype=np.float32),
        "proxy_rewards": np.asarray(proxy_rewards, dtype=np.float32),
        "proxy_improvements": np.asarray(proxy_improvements, dtype=np.float32),
        "floor_reward_flags": np.asarray(floor_reward_flags, dtype=np.float32),
        "step_reward_std": np.asarray(step_reward_std, dtype=np.float32),
        "step_improvement_std": np.asarray(step_improvement_std, dtype=np.float32),
        "same_mask_repeat_std": np.asarray(same_mask_repeat_std, dtype=np.float32),
        "proxy_actual_gain_corr": float(proxy_actual_gain_corr),
        "surrogate_train_times_sec": np.asarray(surrogate_train_times, dtype=np.float32),
        "gfn_train_times_sec": np.asarray(gfn_train_times, dtype=np.float32),
        "proposal_times_sec": np.asarray(proposal_times, dtype=np.float32),
        "iteration_times_sec": np.asarray(iteration_times, dtype=np.float32),
        "total_wall_time_sec": float(perf_counter() - trial_start),
    }
