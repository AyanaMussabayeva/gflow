from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class MaskSpaceSpec:
    hidden_dim: int
    block_size: int
    keep_prob: float

    @property
    def n_blocks_per_layer(self) -> int:
        return self.hidden_dim // self.block_size

    @property
    def total_blocks(self) -> int:
        return 2 * self.n_blocks_per_layer


def sample_random_mask_bits(spec: MaskSpaceSpec, rng: np.random.Generator) -> np.ndarray:
    first = rng.binomial(1, spec.keep_prob, size=spec.n_blocks_per_layer).astype(np.float32)
    second = rng.binomial(1, spec.keep_prob, size=spec.n_blocks_per_layer).astype(np.float32)
    if first.sum() == 0:
        first[int(rng.integers(spec.n_blocks_per_layer))] = 1.0
    if second.sum() == 0:
        second[int(rng.integers(spec.n_blocks_per_layer))] = 1.0
    return np.concatenate([first, second], axis=0)


def expand_mask_bits(spec: MaskSpaceSpec, mask_bits: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    bits = np.asarray(mask_bits, dtype=np.float32)
    bits1 = bits[: spec.n_blocks_per_layer]
    bits2 = bits[spec.n_blocks_per_layer :]
    full1 = np.repeat(bits1, spec.block_size).astype(np.float32)
    full2 = np.repeat(bits2, spec.block_size).astype(np.float32)
    return full1, full2
