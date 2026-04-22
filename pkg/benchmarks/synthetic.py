from __future__ import annotations

from dataclasses import dataclass

import numpy as np


BRANIN_TRUE_MAX = -0.39788735772973816
HARTMANN6_TRUE_MAX = 3.322368011415515
ACKLEY10_TRUE_MAX = -0.0


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
