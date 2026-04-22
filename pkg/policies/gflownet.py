from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import torch
import torch.nn as nn

from pkg.masks import MaskSpaceSpec
from pkg.rewards import RewardSpec, StepContext, evaluate_mask_candidates_with_contexts
from pkg.surrogates import SurrogateBundle


class ContextualMaskGFlowNet(nn.Module):
    def __init__(self, total_blocks: int, context_dim: int, hidden_size: int):
        super().__init__()
        self.total_blocks = total_blocks
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
        logits = self.policy(torch.cat([partial_mask, step_feature, context], dim=1))
        return logits.squeeze(0).squeeze(0)

    def log_z(self, context: torch.Tensor) -> torch.Tensor:
        return self.logz_head(context).squeeze(0).squeeze(0)

    def sample_trajectory(
        self,
        context: np.ndarray,
        mask_space: MaskSpaceSpec,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, torch.Tensor]:
        context_tensor = torch.tensor(context[None, :], dtype=torch.float32)
        state = torch.full((1, self.total_blocks), -1.0, dtype=torch.float32)
        chosen = []
        log_pf_terms = []
        for step in range(self.total_blocks):
            force_keep = False
            if step == mask_space.n_blocks_per_layer - 1 and sum(chosen[:step]) == 0:
                force_keep = True
            if step == self.total_blocks - 1 and sum(chosen[mask_space.n_blocks_per_layer : step]) == 0:
                force_keep = True
            if force_keep:
                action = 1.0
            else:
                logit = self.forward_logit(state, step, context_tensor)
                prob = torch.sigmoid(logit)
                action = float(rng.random() < float(prob.item()))
                log_pf_terms.append(
                    torch.log(prob + 1e-8) if action == 1.0 else torch.log(1.0 - prob + 1e-8)
                )
            chosen.append(action)
            state[0, step] = action
        mask_bits = np.asarray(chosen, dtype=np.float32)
        log_pf = torch.stack(log_pf_terms).sum() if log_pf_terms else torch.tensor(0.0, dtype=torch.float32)
        return mask_bits, log_pf


@dataclass
class GFlowNetTrainStats:
    losses: list[float]
    mean_rewards: list[float]


def train_contextual_gflownet(
    bundle: SurrogateBundle,
    context: np.ndarray,
    y_best: float,
    reward_spec: RewardSpec,
    mask_samples_per_step: int,
    n_steps: int,
    batch_size: int,
    hidden_size: int,
    lr: float,
    rng: np.random.Generator,
    step_context_builder: Callable[[int, int], list[StepContext]],
    gfn: Optional[ContextualMaskGFlowNet] = None,
) -> tuple[ContextualMaskGFlowNet, GFlowNetTrainStats]:
    if gfn is None:
        gfn = ContextualMaskGFlowNet(
            total_blocks=bundle.mask_space.total_blocks,
            context_dim=len(context),
            hidden_size=hidden_size,
        )
    optimizer = torch.optim.Adam(gfn.parameters(), lr=lr)
    context_tensor = torch.tensor(context[None, :], dtype=torch.float32)
    losses = []
    mean_rewards = []
    for step_idx in range(n_steps):
        sampled_masks = []
        sampled_log_pf = []
        for _ in range(batch_size):
            mask_bits, log_pf = gfn.sample_trajectory(context, bundle.mask_space, rng)
            sampled_masks.append(mask_bits)
            sampled_log_pf.append(log_pf)
        evaluations = evaluate_mask_candidates_with_contexts(
            bundle=bundle,
            step_contexts=step_context_builder(batch_size, step_idx),
            mask_bits_list=sampled_masks,
            y_best=y_best,
            reward_spec=reward_spec,
        )
        log_pf_tensor = torch.stack(sampled_log_pf)
        log_r_tensor = torch.log(
            torch.tensor([evaluation.reward for evaluation in evaluations], dtype=torch.float32)
        )
        loss = ((gfn.log_z(context_tensor) + log_pf_tensor - log_r_tensor) ** 2).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(float(loss.item()))
        mean_rewards.append(float(np.mean([evaluation.reward for evaluation in evaluations])))
        _ = mask_samples_per_step
    return gfn, GFlowNetTrainStats(losses=losses, mean_rewards=mean_rewards)
