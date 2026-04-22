from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError:  # pragma: no cover - plotting is optional for smoke tests
    plt = None


@dataclass(frozen=True)
class Benchmark:
    name: str
    dim: int
    lower: np.ndarray
    upper: np.ndarray
    true_max: float

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        if self.name == "branin":
            return branin(x)
        if self.name == "hartmann6":
            return hartmann6(x)
        if self.name == "ackley10":
            return ackley10(x)
        raise ValueError(f"Unknown benchmark: {self.name}")


@dataclass
class ExperimentConfig:
    n_init: int
    n_iter: int
    seeds: List[int]
    surrogate_hidden_dim: int = 64
    surrogate_dropout_p: float = 0.1
    surrogate_epochs: int = 300
    surrogate_lr: float = 1e-3
    block_size: int = 8
    n_candidates: int = 2048
    heldout_mask_samples: int = 24
    random_mask_samples: int = 96
    gfn_mask_samples: int = 96
    gfn_hidden_size: int = 128
    gfn_steps: int = 160
    gfn_batch_size: int = 16
    gfn_lr: float = 1e-3
    proxy_beta: float = 0.6
    reward_temperature: float = 0.5
    continual_finetune: bool = True


BRANIN_TRUE_MAX = -0.39788735772973816
HARTMANN6_TRUE_MAX = 3.322368011415515
ACKLEY10_TRUE_MAX = -0.0


def branin(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    x1 = x[:, 0]
    x2 = x[:, 1]

    a = 1.0
    b = 5.1 / (4.0 * np.pi**2)
    c = 5.0 / np.pi
    r = 6.0
    s = 10.0
    t = 1.0 / (8.0 * np.pi)
    y = a * (x2 - b * x1**2 + c * x1 - r) ** 2 + s * (1.0 - t) * np.cos(x1) + s
    return -y.astype(np.float32)


def hartmann6(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    alpha = np.array([1.0, 1.2, 3.0, 3.2], dtype=np.float32)
    a = np.array(
        [
            [10.0, 3.0, 17.0, 3.5, 1.7, 8.0],
            [0.05, 10.0, 17.0, 0.1, 8.0, 14.0],
            [3.0, 3.5, 1.7, 10.0, 17.0, 8.0],
            [17.0, 8.0, 0.05, 10.0, 0.1, 14.0],
        ],
        dtype=np.float32,
    )
    p = 1e-4 * np.array(
        [
            [1312, 1696, 5569, 124, 8283, 5886],
            [2329, 4135, 8307, 3736, 1004, 9991],
            [2348, 1451, 3522, 2883, 3047, 6650],
            [4047, 8828, 8732, 5743, 1091, 381],
        ],
        dtype=np.float32,
    )

    inner = np.sum(a[None, :, :] * (x[:, None, :] - p[None, :, :]) ** 2, axis=2)
    values = np.sum(alpha[None, :] * np.exp(-inner), axis=1)
    return values.astype(np.float32)


def ackley10(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    dim = x.shape[1]
    squared_norm = np.mean(x**2, axis=1)
    cosine_term = np.mean(np.cos(2.0 * np.pi * x), axis=1)
    value = (
        -20.0 * np.exp(-0.2 * np.sqrt(squared_norm))
        - np.exp(cosine_term)
        + 20.0
        + np.e
    )
    return (-value).astype(np.float32)


def get_benchmark(name: str) -> Benchmark:
    if name == "branin":
        return Benchmark(
            name="branin",
            dim=2,
            lower=np.array([-5.0, 0.0], dtype=np.float32),
            upper=np.array([10.0, 15.0], dtype=np.float32),
            true_max=BRANIN_TRUE_MAX,
        )
    if name == "hartmann6":
        return Benchmark(
            name="hartmann6",
            dim=6,
            lower=np.zeros(6, dtype=np.float32),
            upper=np.ones(6, dtype=np.float32),
            true_max=HARTMANN6_TRUE_MAX,
        )
    if name == "ackley10":
        return Benchmark(
            name="ackley10",
            dim=10,
            lower=-32.768 * np.ones(10, dtype=np.float32),
            upper=32.768 * np.ones(10, dtype=np.float32),
            true_max=ACKLEY10_TRUE_MAX,
        )
    raise ValueError(f"Unsupported benchmark: {name}")


def default_config(benchmark_name: str, seeds: Optional[List[int]] = None) -> ExperimentConfig:
    seeds = list(range(5)) if seeds is None else list(seeds)
    if benchmark_name == "branin":
        return ExperimentConfig(
            n_init=8,
            n_iter=18,
            seeds=seeds,
            surrogate_hidden_dim=64,
            n_candidates=2048,
            gfn_steps=160,
            gfn_batch_size=16,
        )
    if benchmark_name == "hartmann6":
        return ExperimentConfig(
            n_init=14,
            n_iter=22,
            seeds=seeds,
            surrogate_hidden_dim=96,
            n_candidates=4096,
            gfn_steps=180,
            gfn_batch_size=18,
        )
    if benchmark_name == "ackley10":
        return ExperimentConfig(
            n_init=20,
            n_iter=24,
            seeds=seeds,
            surrogate_hidden_dim=128,
            n_candidates=4096,
            gfn_steps=200,
            gfn_batch_size=20,
        )
    raise ValueError(f"Unsupported benchmark: {benchmark_name}")


def set_global_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def sample_uniform(benchmark: Benchmark, n: int, rng: np.random.Generator) -> np.ndarray:
    return benchmark.lower + (benchmark.upper - benchmark.lower) * rng.random((n, benchmark.dim), dtype=np.float32)


class Standardizer:
    def __init__(self, benchmark: Benchmark, y: np.ndarray):
        self.lower = benchmark.lower.astype(np.float32)
        self.upper = benchmark.upper.astype(np.float32)
        self.y_mean = float(np.mean(y))
        self.y_std = float(np.std(y) + 1e-6)

    def normalize_x(self, x: np.ndarray) -> np.ndarray:
        return ((x - self.lower) / (self.upper - self.lower + 1e-8)).astype(np.float32)

    def normalize_y(self, y: np.ndarray) -> np.ndarray:
        return ((y - self.y_mean) / self.y_std).astype(np.float32)

    def denormalize_y(self, y: np.ndarray) -> np.ndarray:
        return (y * self.y_std + self.y_mean).astype(np.float32)


class MaskedMLPSurrogate(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64, dropout_p: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        self.dropout_p = dropout_p
        self.hidden_dim = hidden_dim

    def forward(self, x: torch.Tensor, masks: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> torch.Tensor:
        h = F.relu(self.fc1(x))
        if masks is None:
            h = F.dropout(h, p=self.dropout_p, training=True)
        else:
            mask1, mask2 = masks
            if mask1.ndim == 1:
                mask1 = mask1.unsqueeze(0)
            h = h * mask1

        h = F.relu(self.fc2(h))
        if masks is None:
            h = F.dropout(h, p=self.dropout_p, training=True)
        else:
            if mask2.ndim == 1:
                mask2 = mask2.unsqueeze(0)
            h = h * mask2

        return self.fc3(h).squeeze(-1)


@dataclass
class SurrogateBundle:
    benchmark: Benchmark
    model: MaskedMLPSurrogate
    standardizer: Standardizer
    hidden_dim: int
    block_size: int

    @property
    def n_blocks_per_layer(self) -> int:
        return self.hidden_dim // self.block_size

    @property
    def total_blocks(self) -> int:
        return 2 * self.n_blocks_per_layer

    def split_mask_bits(self, mask_bits: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
        bits = np.asarray(mask_bits, dtype=np.float32)
        bits1 = bits[: self.n_blocks_per_layer]
        bits2 = bits[self.n_blocks_per_layer :]
        full1 = np.repeat(bits1, self.block_size).astype(np.float32)
        full2 = np.repeat(bits2, self.block_size).astype(np.float32)
        mask1 = torch.tensor(full1, dtype=torch.float32)
        mask2 = torch.tensor(full2, dtype=torch.float32)
        return mask1, mask2

    @torch.no_grad()
    def predict_masked(self, x: np.ndarray, mask_bits: np.ndarray) -> np.ndarray:
        x_norm = self.standardizer.normalize_x(x)
        x_tensor = torch.tensor(x_norm, dtype=torch.float32)
        mask1, mask2 = self.split_mask_bits(mask_bits)
        pred = self.model(x_tensor, masks=(mask1, mask2)).cpu().numpy()
        return self.standardizer.denormalize_y(pred)

    @torch.no_grad()
    def mc_predict(self, x: np.ndarray, n_samples: int = 32) -> Tuple[np.ndarray, np.ndarray]:
        x_norm = self.standardizer.normalize_x(x)
        x_tensor = torch.tensor(x_norm, dtype=torch.float32)
        preds = []
        for _ in range(n_samples):
            pred = self.model(x_tensor, masks=None).cpu().numpy()
            preds.append(pred)
        preds = np.stack(preds, axis=0)
        preds = self.standardizer.denormalize_y(preds)
        return preds.mean(axis=0), preds.std(axis=0)


def train_surrogate(
    benchmark: Benchmark,
    x: np.ndarray,
    y: np.ndarray,
    hidden_dim: int,
    dropout_p: float,
    block_size: int,
    epochs: int,
    lr: float,
) -> SurrogateBundle:
    if hidden_dim % block_size != 0:
        raise ValueError("hidden_dim must be divisible by block_size.")

    standardizer = Standardizer(benchmark, y)
    x_norm = standardizer.normalize_x(x)
    y_norm = standardizer.normalize_y(y)

    x_tensor = torch.tensor(x_norm, dtype=torch.float32)
    y_tensor = torch.tensor(y_norm, dtype=torch.float32)

    model = MaskedMLPSurrogate(
        input_dim=benchmark.dim,
        hidden_dim=hidden_dim,
        dropout_p=dropout_p,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()
    for _ in range(epochs):
        pred = model(x_tensor, masks=None)
        loss = F.mse_loss(pred, y_tensor)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    model.eval()
    return SurrogateBundle(
        benchmark=benchmark,
        model=model,
        standardizer=standardizer,
        hidden_dim=hidden_dim,
        block_size=block_size,
    )


def sample_random_mask_bits(bundle: SurrogateBundle, rng: np.random.Generator) -> np.ndarray:
    first = rng.binomial(1, 0.5, size=bundle.n_blocks_per_layer).astype(np.float32)
    second = rng.binomial(1, 0.5, size=bundle.n_blocks_per_layer).astype(np.float32)

    if first.sum() == 0:
        first[rng.integers(bundle.n_blocks_per_layer)] = 1.0
    if second.sum() == 0:
        second[rng.integers(bundle.n_blocks_per_layer)] = 1.0
    return np.concatenate([first, second], axis=0)


def dataset_context(
    benchmark: Benchmark,
    x: np.ndarray,
    y: np.ndarray,
    standardizer: Standardizer,
    n_init: int,
    n_iter: int,
) -> np.ndarray:
    x_norm = standardizer.normalize_x(x)
    y_norm = standardizer.normalize_y(y)
    progress = (len(y) - n_init) / max(n_iter, 1)
    x_dispersion = float(np.mean(np.var(x_norm, axis=0)))
    ctx = np.array(
        [
            float(progress),
            float(np.max(y_norm)),
            float(np.mean(y_norm)),
            float(np.std(y_norm)),
            x_dispersion,
            benchmark.dim / 10.0,
        ],
        dtype=np.float32,
    )
    return ctx


class ContextualMaskGFlowNet(nn.Module):
    def __init__(self, total_blocks: int, context_dim: int, hidden_size: int = 128):
        super().__init__()
        self.total_blocks = total_blocks
        self.context_dim = context_dim
        self.policy = nn.Sequential(
            nn.Linear(total_blocks + 1 + context_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )
        self.logz_head = nn.Sequential(
            nn.Linear(context_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    def forward_logit(self, partial_mask: torch.Tensor, step_idx: int, context: torch.Tensor) -> torch.Tensor:
        step_feature = torch.tensor([[step_idx / self.total_blocks]], dtype=torch.float32)
        inputs = torch.cat([partial_mask, step_feature, context], dim=1)
        return self.policy(inputs).squeeze(0).squeeze(0)

    def log_z(self, context: torch.Tensor) -> torch.Tensor:
        return self.logz_head(context).squeeze(0).squeeze(0)

    def sample_trajectory(
        self,
        context: np.ndarray,
        n_blocks_per_layer: int,
        rng: np.random.Generator,
        greedy: bool = False,
    ) -> Tuple[np.ndarray, torch.Tensor]:
        context_tensor = torch.tensor(context[None, :], dtype=torch.float32)
        state = torch.full((1, self.total_blocks), -1.0, dtype=torch.float32)
        chosen: List[float] = []
        log_pf_terms: List[torch.Tensor] = []

        for step in range(self.total_blocks):
            force_keep = False
            if step == n_blocks_per_layer - 1 and sum(chosen[:step]) == 0:
                force_keep = True
            if step == self.total_blocks - 1 and sum(chosen[n_blocks_per_layer:step]) == 0:
                force_keep = True

            if force_keep:
                action = 1.0
            else:
                logit = self.forward_logit(state, step, context_tensor)
                prob = torch.sigmoid(logit)
                if greedy:
                    action = float(prob.item() >= 0.5)
                else:
                    action = float(rng.random() < float(prob.item()))
                log_prob = torch.log(prob + 1e-8) if action == 1.0 else torch.log(1.0 - prob + 1e-8)
                log_pf_terms.append(log_prob)

            chosen.append(action)
            state[0, step] = action

        mask_bits = np.asarray(chosen, dtype=np.float32)
        if log_pf_terms:
            return mask_bits, torch.stack(log_pf_terms).sum()
        return mask_bits, torch.tensor(0.0, dtype=torch.float32)


def proxy_reward_for_mask(
    bundle: SurrogateBundle,
    mask_bits: np.ndarray,
    y_best: float,
    rng: np.random.Generator,
    n_candidates: int,
    heldout_mask_samples: int,
    proxy_beta: float,
    reward_temperature: float,
) -> Dict[str, np.ndarray | float]:
    candidates = sample_uniform(bundle.benchmark, n_candidates, rng)
    masked_preds = bundle.predict_masked(candidates, mask_bits)
    best_idx = int(np.argmax(masked_preds))
    x_best = candidates[best_idx : best_idx + 1]
    masked_pred = float(masked_preds[best_idx])

    heldout_preds = []
    for _ in range(heldout_mask_samples):
        heldout_bits = sample_random_mask_bits(bundle, rng)
        heldout_pred = float(bundle.predict_masked(x_best, heldout_bits)[0])
        heldout_preds.append(heldout_pred)
    heldout_preds = np.asarray(heldout_preds, dtype=np.float32)

    proxy_mean = float(np.mean(heldout_preds))
    proxy_std = float(np.std(heldout_preds))
    improvement = proxy_mean + proxy_beta * proxy_std - float(y_best)
    scaled_improvement = improvement / max(reward_temperature, 1e-6)
    reward = float(F.softplus(torch.tensor(scaled_improvement)).item() + 1e-4)

    return {
        "reward": reward,
        "x_next": x_best,
        "masked_pred": masked_pred,
        "proxy_mean": proxy_mean,
        "proxy_std": proxy_std,
        "improvement": improvement,
    }


def train_contextual_gflownet(
    bundle: SurrogateBundle,
    context: np.ndarray,
    y_best: float,
    cfg: ExperimentConfig,
    rng: np.random.Generator,
    gfn: Optional[ContextualMaskGFlowNet] = None,
) -> Tuple[ContextualMaskGFlowNet, Dict[str, List[float]]]:
    if gfn is None:
        gfn = ContextualMaskGFlowNet(
            total_blocks=bundle.total_blocks,
            context_dim=len(context),
            hidden_size=cfg.gfn_hidden_size,
        )

    optimizer = torch.optim.Adam(gfn.parameters(), lr=cfg.gfn_lr)
    losses: List[float] = []
    rewards: List[float] = []

    context_tensor = torch.tensor(context[None, :], dtype=torch.float32)
    for _ in range(cfg.gfn_steps):
        batch_log_pf = []
        batch_log_r = []
        batch_rewards = []

        for _ in range(cfg.gfn_batch_size):
            mask_bits, log_pf = gfn.sample_trajectory(
                context=context,
                n_blocks_per_layer=bundle.n_blocks_per_layer,
                rng=rng,
                greedy=False,
            )
            reward_info = proxy_reward_for_mask(
                bundle=bundle,
                mask_bits=mask_bits,
                y_best=y_best,
                rng=rng,
                n_candidates=cfg.n_candidates,
                heldout_mask_samples=cfg.heldout_mask_samples,
                proxy_beta=cfg.proxy_beta,
                reward_temperature=cfg.reward_temperature,
            )
            batch_log_pf.append(log_pf)
            batch_log_r.append(torch.log(torch.tensor(reward_info["reward"], dtype=torch.float32)))
            batch_rewards.append(float(reward_info["reward"]))

        log_pf_tensor = torch.stack(batch_log_pf)
        log_r_tensor = torch.stack(batch_log_r)
        log_z = gfn.log_z(context_tensor)
        loss = ((log_z + log_pf_tensor - log_r_tensor) ** 2).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(float(loss.item()))
        rewards.append(float(np.mean(batch_rewards)))

    return gfn, {"loss": losses, "mean_reward": rewards}


def select_next_with_random_masks(
    bundle: SurrogateBundle,
    y_best: float,
    cfg: ExperimentConfig,
    rng: np.random.Generator,
) -> Dict[str, np.ndarray | float]:
    best_info = None
    for _ in range(cfg.random_mask_samples):
        mask_bits = sample_random_mask_bits(bundle, rng)
        info = proxy_reward_for_mask(
            bundle=bundle,
            mask_bits=mask_bits,
            y_best=y_best,
            rng=rng,
            n_candidates=cfg.n_candidates,
            heldout_mask_samples=cfg.heldout_mask_samples,
            proxy_beta=cfg.proxy_beta,
            reward_temperature=cfg.reward_temperature,
        )
        info["mask_bits"] = mask_bits
        if best_info is None or float(info["reward"]) > float(best_info["reward"]):
            best_info = info
    assert best_info is not None
    return best_info


def select_next_with_gfn(
    bundle: SurrogateBundle,
    gfn: ContextualMaskGFlowNet,
    context: np.ndarray,
    y_best: float,
    cfg: ExperimentConfig,
    rng: np.random.Generator,
) -> Dict[str, np.ndarray | float]:
    best_info = None
    for _ in range(cfg.gfn_mask_samples):
        mask_bits, _ = gfn.sample_trajectory(
            context=context,
            n_blocks_per_layer=bundle.n_blocks_per_layer,
            rng=rng,
            greedy=False,
        )
        info = proxy_reward_for_mask(
            bundle=bundle,
            mask_bits=mask_bits,
            y_best=y_best,
            rng=rng,
            n_candidates=cfg.n_candidates,
            heldout_mask_samples=cfg.heldout_mask_samples,
            proxy_beta=cfg.proxy_beta,
            reward_temperature=cfg.reward_temperature,
        )
        info["mask_bits"] = mask_bits
        if best_info is None or float(info["reward"]) > float(best_info["reward"]):
            best_info = info
    assert best_info is not None
    return best_info


def run_single_trial(
    benchmark_name: str,
    method: str,
    seed: int,
    cfg: Optional[ExperimentConfig] = None,
    collect_timing: bool = False,
    verbose: bool = False,
    progress_label: Optional[str] = None,
) -> Dict[str, np.ndarray | float | List[float]]:
    benchmark = get_benchmark(benchmark_name)
    cfg = default_config(benchmark_name, [seed]) if cfg is None else cfg
    rng = np.random.default_rng(seed)
    set_global_seed(seed)

    x = sample_uniform(benchmark, cfg.n_init, rng)
    y = benchmark.evaluate(x)

    regrets: List[float] = []
    best_values: List[float] = []
    queried_values: List[float] = []
    proxy_rewards: List[float] = []
    proxy_improvements: List[float] = []
    gfn_training_rewards: List[float] = []
    gfn_training_losses: List[float] = []
    surrogate_train_times: List[float] = []
    gfn_train_times: List[float] = []
    proposal_times: List[float] = []
    oracle_eval_times: List[float] = []
    iteration_times: List[float] = []

    gfn: Optional[ContextualMaskGFlowNet] = None
    trial_start = perf_counter()

    label = progress_label or f"{benchmark_name}:{method}:seed={seed}"

    for iter_idx in range(cfg.n_iter):
        iter_start = perf_counter()
        surrogate_start = perf_counter()
        bundle = train_surrogate(
            benchmark=benchmark,
            x=x,
            y=y,
            hidden_dim=cfg.surrogate_hidden_dim,
            dropout_p=cfg.surrogate_dropout_p,
            block_size=cfg.block_size,
            epochs=cfg.surrogate_epochs,
            lr=cfg.surrogate_lr,
        )
        surrogate_train_times.append(perf_counter() - surrogate_start)
        context = dataset_context(
            benchmark=benchmark,
            x=x,
            y=y,
            standardizer=bundle.standardizer,
            n_init=cfg.n_init,
            n_iter=cfg.n_iter,
        )
        y_best = float(np.max(y))

        if method == "random":
            proposal_start = perf_counter()
            proposal = select_next_with_random_masks(bundle, y_best, cfg, rng)
            proposal_times.append(perf_counter() - proposal_start)
            if collect_timing:
                gfn_train_times.append(0.0)
        elif method == "gfn":
            if not cfg.continual_finetune:
                gfn = None
            gfn_start = perf_counter()
            gfn, train_stats = train_contextual_gflownet(
                bundle=bundle,
                context=context,
                y_best=y_best,
                cfg=cfg,
                rng=rng,
                gfn=gfn,
            )
            gfn_train_times.append(perf_counter() - gfn_start)
            gfn_training_losses.append(train_stats["loss"][-1])
            gfn_training_rewards.append(train_stats["mean_reward"][-1])
            proposal_start = perf_counter()
            proposal = select_next_with_gfn(bundle, gfn, context, y_best, cfg, rng)
            proposal_times.append(perf_counter() - proposal_start)
        else:
            raise ValueError("method must be 'random' or 'gfn'")

        x_next = np.asarray(proposal["x_next"], dtype=np.float32)
        oracle_start = perf_counter()
        y_next = benchmark.evaluate(x_next)
        oracle_eval_times.append(perf_counter() - oracle_start)

        x = np.concatenate([x, x_next], axis=0)
        y = np.concatenate([y, y_next], axis=0)

        best_so_far = float(np.max(y))
        regret = float(benchmark.true_max - best_so_far)

        queried_values.append(float(y_next[0]))
        best_values.append(best_so_far)
        regrets.append(regret)
        proxy_rewards.append(float(proposal["reward"]))
        proxy_improvements.append(float(proposal["improvement"]))
        iteration_times.append(perf_counter() - iter_start)

        if verbose:
            elapsed = perf_counter() - trial_start
            bar_width = 24
            filled = int(bar_width * (iter_idx + 1) / cfg.n_iter)
            bar = "#" * filled + "-" * (bar_width - filled)
            print(
                f"[{label}] [{bar}] {iter_idx + 1}/{cfg.n_iter} "
                f"elapsed={elapsed:.1f}s best={best_so_far:.4f} regret={regret:.4f}",
                flush=True,
            )

    total_wall_time = perf_counter() - trial_start

    result = {
        "seed": seed,
        "method": method,
        "benchmark": benchmark_name,
        "regrets": np.asarray(regrets, dtype=np.float32),
        "best_values": np.asarray(best_values, dtype=np.float32),
        "queried_values": np.asarray(queried_values, dtype=np.float32),
        "proxy_rewards": np.asarray(proxy_rewards, dtype=np.float32),
        "proxy_improvements": np.asarray(proxy_improvements, dtype=np.float32),
        "gfn_training_rewards": np.asarray(gfn_training_rewards, dtype=np.float32),
        "gfn_training_losses": np.asarray(gfn_training_losses, dtype=np.float32),
    }
    if collect_timing:
        result.update(
            {
                "iteration_times_sec": np.asarray(iteration_times, dtype=np.float32),
                "surrogate_train_times_sec": np.asarray(surrogate_train_times, dtype=np.float32),
                "gfn_train_times_sec": np.asarray(gfn_train_times, dtype=np.float32),
                "proposal_times_sec": np.asarray(proposal_times, dtype=np.float32),
                "oracle_eval_times_sec": np.asarray(oracle_eval_times, dtype=np.float32),
                "total_wall_time_sec": float(total_wall_time),
            }
        )
    return result


def summarize_trials(trials: List[Dict[str, np.ndarray | float | List[float]]]) -> Dict[str, np.ndarray | float]:
    regrets = np.stack([trial["regrets"] for trial in trials], axis=0)
    best_values = np.stack([trial["best_values"] for trial in trials], axis=0)
    queried_values = np.stack([trial["queried_values"] for trial in trials], axis=0)

    return {
        "mean_regret": regrets.mean(axis=0),
        "std_regret": regrets.std(axis=0),
        "mean_best_value": best_values.mean(axis=0),
        "std_best_value": best_values.std(axis=0),
        "mean_query_value": queried_values.mean(axis=0),
        "std_query_value": queried_values.std(axis=0),
        "final_regret_mean": float(regrets[:, -1].mean()),
        "final_regret_std": float(regrets[:, -1].std()),
        "final_best_mean": float(best_values[:, -1].mean()),
        "final_best_std": float(best_values[:, -1].std()),
    }


def run_benchmark_comparison(
    benchmark_name: str,
    cfg: Optional[ExperimentConfig] = None,
) -> Dict[str, Dict[str, np.ndarray | float | List[Dict[str, np.ndarray | float | List[float]]]]]:
    cfg = default_config(benchmark_name) if cfg is None else cfg
    random_trials = [run_single_trial(benchmark_name, "random", seed, cfg) for seed in cfg.seeds]
    gfn_trials = [run_single_trial(benchmark_name, "gfn", seed, cfg) for seed in cfg.seeds]

    return {
        "random": {
            "trials": random_trials,
            "summary": summarize_trials(random_trials),
        },
        "gfn": {
            "trials": gfn_trials,
            "summary": summarize_trials(gfn_trials),
        },
    }


def comparison_table(results: Dict[str, Dict[str, Dict[str, np.ndarray | float]]]) -> Dict[str, Dict[str, float]]:
    return {
        "random": {
            "final_regret_mean": float(results["random"]["summary"]["final_regret_mean"]),
            "final_regret_std": float(results["random"]["summary"]["final_regret_std"]),
            "final_best_mean": float(results["random"]["summary"]["final_best_mean"]),
            "final_best_std": float(results["random"]["summary"]["final_best_std"]),
        },
        "gfn": {
            "final_regret_mean": float(results["gfn"]["summary"]["final_regret_mean"]),
            "final_regret_std": float(results["gfn"]["summary"]["final_regret_std"]),
            "final_best_mean": float(results["gfn"]["summary"]["final_best_mean"]),
            "final_best_std": float(results["gfn"]["summary"]["final_best_std"]),
        },
    }


def print_comparison_table(results: Dict[str, Dict[str, Dict[str, np.ndarray | float]]]) -> None:
    table = comparison_table(results)
    print("Method        Final regret (mean +- std)    Final best value (mean +- std)")
    for method, stats in table.items():
        print(
            f"{method:<12}"
            f"{stats['final_regret_mean']:.4f} +- {stats['final_regret_std']:.4f}        "
            f"{stats['final_best_mean']:.4f} +- {stats['final_best_std']:.4f}"
        )


def plot_regret_comparison(
    results: Dict[str, Dict[str, Dict[str, np.ndarray | float]]],
    title: str,
) -> None:
    if plt is None:
        raise ModuleNotFoundError("matplotlib is required for plotting.")
    x_axis = np.arange(1, len(results["random"]["summary"]["mean_regret"]) + 1)
    plt.figure(figsize=(8, 5))

    for method, color in [("random", "tab:blue"), ("gfn", "tab:orange")]:
        mean = results[method]["summary"]["mean_regret"]
        std = results[method]["summary"]["std_regret"]
        label = "Random mask sampling" if method == "random" else "Contextual GFlowNet masks"
        plt.plot(x_axis, mean, label=label, color=color)
        plt.fill_between(x_axis, mean - std, mean + std, alpha=0.2, color=color)

    plt.xlabel("BO iteration")
    plt.ylabel("Simple regret")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()


def plot_proxy_diagnostics(trial: Dict[str, np.ndarray | float | List[float]], title: str) -> None:
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
