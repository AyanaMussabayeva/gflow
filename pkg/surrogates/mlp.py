from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from pkg.benchmarks import Benchmark
from pkg.masks import MaskSpaceSpec, expand_mask_bits, sample_random_mask_bits


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
    def __init__(self, input_dim: int, hidden_dim: int, dropout_p: float):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        self.dropout_p = dropout_p

    def forward(
        self,
        x: torch.Tensor,
        masks: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
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
    mask_space: MaskSpaceSpec

    @torch.no_grad()
    def predict_masked(self, x: np.ndarray, mask_bits: np.ndarray) -> np.ndarray:
        x_norm = self.standardizer.normalize_x(x)
        x_tensor = torch.tensor(x_norm, dtype=torch.float32)
        mask1, mask2 = expand_mask_bits(self.mask_space, mask_bits)
        mask1_tensor = torch.tensor(mask1, dtype=torch.float32)
        mask2_tensor = torch.tensor(mask2, dtype=torch.float32)
        pred = self.model(x_tensor, masks=(mask1_tensor, mask2_tensor)).cpu().numpy()
        return self.standardizer.denormalize_y(pred)

    @torch.no_grad()
    def mc_predict(self, x: np.ndarray, n_samples: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        preds = []
        for _ in range(n_samples):
            mask_bits = sample_random_mask_bits(self.mask_space, rng)
            preds.append(self.predict_masked(x, mask_bits))
        pred_stack = np.stack(preds, axis=0)
        return pred_stack.mean(axis=0), pred_stack.std(axis=0)


def train_surrogate(
    benchmark: Benchmark,
    x: np.ndarray,
    y: np.ndarray,
    mask_space: MaskSpaceSpec,
    dropout_p: float,
    epochs: int,
    lr: float,
    rng: np.random.Generator,
    use_block_mask_training: bool,
) -> SurrogateBundle:
    if mask_space.hidden_dim % mask_space.block_size != 0:
        raise ValueError("hidden_dim must be divisible by block_size.")

    standardizer = Standardizer(benchmark, y)
    x_norm = standardizer.normalize_x(x)
    y_norm = standardizer.normalize_y(y)
    x_tensor = torch.tensor(x_norm, dtype=torch.float32)
    y_tensor = torch.tensor(y_norm, dtype=torch.float32)

    model = MaskedMLPSurrogate(
        input_dim=benchmark.dim,
        hidden_dim=mask_space.hidden_dim,
        dropout_p=dropout_p,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    for _ in range(epochs):
        if use_block_mask_training:
            mask_bits = sample_random_mask_bits(mask_space, rng)
            mask1, mask2 = expand_mask_bits(mask_space, mask_bits)
            pred = model(
                x_tensor,
                masks=(
                    torch.tensor(mask1, dtype=torch.float32),
                    torch.tensor(mask2, dtype=torch.float32),
                ),
            )
        else:
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
        mask_space=mask_space,
    )
