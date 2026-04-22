from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import torch


@dataclass
class SeedStreams:
    base_seed: int
    _cache: Dict[str, np.random.Generator]

    def __init__(self, base_seed: int):
        self.base_seed = int(base_seed)
        self._cache = {}

    def generator(self, name: str) -> np.random.Generator:
        if name not in self._cache:
            seed = abs(hash((self.base_seed, name))) % (2**32)
            self._cache[name] = np.random.default_rng(seed)
        return self._cache[name]


def set_global_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)

