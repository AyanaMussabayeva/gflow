from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class RewardSpec:
    protocol: str = "rank"
    proxy_beta: float = 0.6
    reward_temperature: float = 0.5
    epsilon: float = 1e-4

    def reward_from_improvements(self, improvement: float, all_improvements: np.ndarray) -> float:
        if self.protocol == "raw":
            return float(max(improvement, 0.0) + self.epsilon)
        if self.protocol == "softplus_scaled":
            scaled = improvement / max(self.reward_temperature, 1e-6)
            return float(F.softplus(torch.tensor(scaled)).item() + self.epsilon)
        if self.protocol == "zscore":
            mean = float(np.mean(all_improvements))
            std = float(np.std(all_improvements) + 1e-6)
            zscore = (improvement - mean) / std
            return float(F.softplus(torch.tensor(zscore)).item() + self.epsilon)
        if self.protocol == "rank":
            less_equal = float(np.mean(all_improvements <= improvement))
            return float(max(less_equal, self.epsilon))
        raise ValueError(f"Unsupported reward protocol: {self.protocol}")
