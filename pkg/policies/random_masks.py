from __future__ import annotations

import numpy as np

from pkg.masks import MaskSpaceSpec, sample_random_mask_bits


def sample_random_policy_masks(
    mask_space: MaskSpaceSpec,
    n_masks: int,
    rng: np.random.Generator,
) -> list[np.ndarray]:
    return [sample_random_mask_bits(mask_space, rng) for _ in range(n_masks)]
